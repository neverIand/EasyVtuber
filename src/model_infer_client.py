import os
from multiprocessing import Process, Value, shared_memory, Event
import cv2
import numpy as np
import time
from .ezvtb_rt_interface import get_core
from .args import args
from .utils.shared_mem_guard import SharedMemoryGuard
from .utils.pose_simplify import pose_simplify
from .utils.fps import FPS, Interval
from .utils.gpu_duty_limiter import GpuDutyCycleLimiter
from .utils.frame_transform import apply_output_transform, build_output_transform
from typing import List

class ModelClientProcess(Process):
    def __init__(self, input_image, pose_position_shm: shared_memory.SharedMemory , input_fps):
        super().__init__()
        self.input_image = input_image
        self.pose_position_shm = pose_position_shm  # 45 floats for pose, 4 floats for position
        
        self.alpha_width_scale = 2 if args.alpha_split else 1
        self.ret_channels = 3 if args.output_virtual_cam or args.output_debug else 4
        self.ret_shape = (args.interpolation_scale, args.model_output_size, self.alpha_width_scale * args.model_output_size, self.ret_channels)
        self.ret_nbytes = self.alpha_width_scale * args.interpolation_scale * args.model_output_size * args.model_output_size * self.ret_channels # RGBA
        self.ret_shared_mem = shared_memory.SharedMemory(create=True, size=self.ret_nbytes)
        
        self.last_model_interval = Value('f', 0.0)
        self.average_model_interval = Value('f', 0.0)
        self.cache_hit_ratio = Value('f', 0.0)
        self.gpu_cache_hit_ratio = Value('f', 0.0)
        self.pipeline_fps_number = Value('f', 0.0)
        self.output_pipeline_fps = Value('f', 0.0)  # 由 main 进程更新，与 main print 一致
        self.input_fps = input_fps

        self.finish_event = Event()

    def run(self):
        # 插值帧红点标记：由 --mark_interpolated 控制，供 ezvtb_rt 读取
        if getattr(args, 'mark_interpolated', False):
            os.environ['EZVTB_MARK_INTERPOLATED'] = '1'
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        
        ret_batch_shm_guard = [
            SharedMemoryGuard(self.ret_shared_mem, ctrl_name=f"ret_shm_ctrl_batch_{i}")
            for i in range(args.interpolation_scale)
        ]
        np_ret_shms = [
            np.ndarray((args.model_output_size, self.alpha_width_scale * args.model_output_size, self.ret_channels), dtype=np.uint8,
                        buffer=self.ret_shared_mem.buf[i * self.alpha_width_scale * args.model_output_size * args.model_output_size * self.ret_channels:
                                                       (i + 1) * self.alpha_width_scale * args.model_output_size * args.model_output_size * self.ret_channels])
            for i in range(args.interpolation_scale)
        ]
        postprocess_bgra = np.empty(
            (args.model_output_size, args.model_output_size, 4),
            dtype=np.uint8,
        )
        postprocess_rgba = (
            np.empty_like(postprocess_bgra) if args.alpha_split else None
        )

        model_infer_average_interval: Interval = Interval()
        pipeline_fps = FPS()
        gpu_duty_limiter = GpuDutyCycleLimiter(args.gpu_duty_limit)

        # TensorRT engine construction and execution-context JIT happen before
        # the per-frame limiter can measure inference. Pass the same safety
        # target into the TensorRT layer so those indivisible startup calls are
        # followed by a proportional cooldown as well.
        if args.use_tensorrt:
            os.environ['EZVTB_GPU_DUTY_LIMIT'] = str(args.gpu_duty_limit)

        # Use unified ezvtb_rt interface for both THA3 and THA4
        model = get_core(use_tensorrt=args.use_tensorrt,
                            model_version=args.model_version,
                            model_name=args.model_name,

                            model_seperable = args.model_seperable,
                            model_half=args.model_half, 
                            model_cache_size=args.max_gpu_cache_len, 
                            model_use_eyebrow=args.eyebrow,

                            use_interpolation=args.use_interpolation,
                            interpolation_scale=args.interpolation_scale,
                            interpolation_half=args.interpolation_half,

                            cacher_ram_size=args.max_ram_cache_len,
                            cache_storage_mode=args.ram_cache_mode,

                            use_sr=args.use_sr,
                            sr_half=args.sr_half,
                            sr_x4=args.sr_x4,
                            sr_a4k=args.sr_a4k,
                            )
        print(
            'Inference backend: {}'.format(
                'TensorRT' if args.use_tensorrt else 'DirectML'
            )
        )
        model.setImage(self.input_image)
        model_infer_average_interval.start()
        warmup_started_at = time.perf_counter()
        model.inference(
            [np.zeros((1, 45), dtype=np.float32)],
            copy_output=False,
        )  # Warm up
        gpu_duty_limiter.record_inference(warmup_started_at)
        model_infer_average_interval.stop()
        self.last_model_interval.value = model_infer_average_interval.last()
        # Keep the warm-up inside the same sustained-duty budget as the main
        # loop.  This is normally only a few milliseconds at 90%, but avoids
        # immediately stacking the first live frame onto startup JIT work.
        gpu_duty_limiter.wait()

        last_pose = np.zeros((45,), dtype=np.float32)

        print(
            "GPU inference duty limit: {:.1f}% "
            "(sustained target; instantaneous utilization may be higher)".format(
                args.gpu_duty_limit
            )
        )
        print("Model Inference Ready")
        while True:
            with pose_position_shm_guard.lock():
                np_pose = np_pose_shm.copy()
                np_position = np_position_shm.copy()

            input_poses = []
            increment = (np_pose - last_pose) / args.interpolation_scale
            for i in range(args.interpolation_scale):
                input_poses.append(pose_simplify(last_pose + increment * (i + 1)))
            last_pose = np_pose

            gpu_duty_limiter.wait()
            model_infer_average_interval.start()
            inference_started_at = time.perf_counter()
            # Post-processing completes before the next inference, so the
            # TensorRT host buffer can be consumed directly.  Public runtime
            # callers still receive a defensive copy by default.
            output_images = model.inference(input_poses, copy_output=False)
            gpu_duty_limiter.record_inference(inference_started_at)

            if args.max_ram_cache_len > 0:
                hits = model.cacher.hits
                miss = model.cacher.miss
                if args.use_sr:
                    hits += model.sr_cacher.hits
                    miss += model.sr_cacher.miss
                total = hits + miss
                self.cache_hit_ratio.value = (hits / total) if total > 0 else 0.0

            if args.use_tensorrt and args.max_gpu_cache_len > 0:
                hits = model.tha.cacher.hits
                miss = model.tha.cacher.miss
                total = hits + miss
                self.gpu_cache_hit_ratio.value = (hits / total) if total > 0 else 0.0

            self.post_process_into(
                np_position,
                output_images,
                np_ret_shms,
                ret_batch_shm_guard,
                postprocess_bgra,
                postprocess_rgba,
            )

            self.average_model_interval.value = model_infer_average_interval.stop()
            self.last_model_interval.value = model_infer_average_interval.last()

            self.pipeline_fps_number.value = pipeline_fps()
            self.finish_event.set() # Back pressure main process loop if infer slow

    def post_process_into(
        self,
        np_position: np.ndarray,
        output_images: np.ndarray,
        destinations: List[np.ndarray],
        destination_guards: List[SharedMemoryGuard],
        bgra_work: np.ndarray,
        rgba_work: np.ndarray | None,
    ) -> None:
        transform = build_output_transform(
            np_position,
            output_images[0].shape,
            args.extend_movement,
            args.bongo,
        )

        for i in range(output_images.shape[0]):
            # Preserve the existing output back-pressure contract: the model
            # process may not overwrite a shared slot until main releases it.
            with destination_guards[i].lock():
                self._post_process_frame_into(
                    output_images[i],
                    transform,
                    destinations[i],
                    bgra_work,
                    rgba_work,
                )

    def _draw_debug_overlay(self, bgra_image: np.ndarray) -> None:
        # Keep text, order, rounding, and coordinates byte-for-byte compatible
        # with the legacy post-processing path.
        y = 16
        cv2.putText(bgra_image, 'INFER/S: {:.4f}'.format(self.pipeline_fps_number.value), (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        y += 16
        cv2.putText(bgra_image, 'INPUT/S: {:.4f}'.format(self.input_fps.value), (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        y += 16
        cv2.putText(bgra_image, 'OUTPUT/S: {:.4f}'.format(self.output_pipeline_fps.value), (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        y += 16
        cv2.putText(bgra_image, 'CALC: {:.2f} ms'.format(self.average_model_interval.value * 1000), (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        y += 16
        if args.max_ram_cache_len > 0:
            cv2.putText(bgra_image, 'MEM CACHE: {:.2f}%'.format(self.cache_hit_ratio.value * 100), (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            y += 16
        if args.max_gpu_cache_len > 0:
            cv2.putText(bgra_image, 'GPU CACHE: {:.2f}%'.format(self.gpu_cache_hit_ratio.value * 100), (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    def _post_process_frame_into(
        self,
        output_image: np.ndarray,
        transform: np.ndarray | None,
        destination: np.ndarray,
        bgra_work: np.ndarray,
        rgba_work: np.ndarray | None,
    ) -> np.ndarray:
        if transform is None and not args.output_debug:
            bgra_image = output_image
        else:
            bgra_image = apply_output_transform(
                output_image,
                transform,
                dst=bgra_work,
            )

        if args.output_debug:
            self._draw_debug_overlay(bgra_image)

        if args.alpha_split:
            if rgba_work is None:
                raise ValueError('rgba_work is required for alpha-split output')
            cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA, dst=rgba_work)
            width = output_image.shape[1]
            left = destination[:, :width, :]
            right = destination[:, width:, :]
            if args.output_debug:
                cv2.cvtColor(
                    rgba_work[:, :, :3],
                    cv2.COLOR_RGB2BGR,
                    dst=left,
                )
            else:
                np.copyto(left, rgba_work[:, :, :3])
            cv2.cvtColor(
                rgba_work[:, :, 3],
                cv2.COLOR_GRAY2RGB,
                dst=right,
            )
        elif args.output_debug:
            cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR, dst=destination)
        elif args.output_virtual_cam:
            cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGB, dst=destination)
        else:
            cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA, dst=destination)
        return destination

    def post_process_ret(self, np_position: np.ndarray, output_images: np.ndarray) -> List[np.ndarray]:
        """Compatibility wrapper that allocates outputs for external callers.

        The live process uses :meth:`post_process_into` to write shared memory
        directly and reuse both work buffers across frames.
        """
        transform = build_output_transform(
            np_position,
            output_images[0].shape,
            args.extend_movement,
            args.bongo,
        )
        bgra_work = np.empty_like(output_images[0])
        rgba_work = np.empty_like(output_images[0]) if args.alpha_split else None

        ret = []
        for i in range(output_images.shape[0]):
            height, width = output_images[i].shape[:2]
            output_width = width * (2 if args.alpha_split else 1)
            output_channels = 3 if args.output_debug or args.output_virtual_cam else 4
            destination = np.empty(
                (height, output_width, output_channels),
                dtype=np.uint8,
            )
            ret.append(
                self._post_process_frame_into(
                    output_images[i],
                    transform,
                    destination,
                    bgra_work,
                    rgba_work,
                )
            )
        return ret
    
if __name__ == "__main__":
    pass
