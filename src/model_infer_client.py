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
        model.inference([np.zeros((1, 45), dtype=np.float32)])  # Warm up
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
            output_images = model.inference(input_poses)
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

            output_images = self.post_process_ret(np_position, output_images)

            self.average_model_interval.value = model_infer_average_interval.stop()
            self.last_model_interval.value = model_infer_average_interval.last()

            self.pipeline_fps_number.value = pipeline_fps()
            for i in range(args.interpolation_scale):
                with ret_batch_shm_guard[i].lock(): # get pressure from main process if ret not consumed
                    np_ret_shms[i][:, :, :] = output_images[i]

            self.finish_event.set() # Back pressure main process loop if infer slow

    def post_process_ret(self, np_position: np.ndarray, output_images: np.ndarray) -> List[np.ndarray]:
        transform = build_output_transform(
            np_position,
            output_images[0].shape,
            args.extend_movement,
            args.bongo,
        )

        ret = []
        for i in range(output_images.shape[0]):
            # Debug overlays mutate the frame, so preserve the model/cache buffer
            # while still avoiding the much more expensive identity warp.
            bgra_image = apply_output_transform(
                output_images[i],
                transform,
                copy_identity=args.output_debug,
            )

            if args.output_debug:
                # 与 main.py 输出格式一致
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
                
            if args.alpha_split:
                rgba_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA)
                alpha_channel = rgba_image[:, :, 3]
                rgb_channels = rgba_image[:,:,:3]
                alpha_image = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2RGB)
                rgb_channels = cv2.hconcat([rgb_channels, alpha_image])

            if args.output_debug:
                if args.alpha_split:
                    bgr_channels = cv2.cvtColor(rgb_channels, cv2.COLOR_RGB2BGR)
                else:
                    bgr_channels = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)
                ret.append(bgr_channels)
            elif args.output_virtual_cam:
                if not args.alpha_split:
                    rgb_channels = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGB)
                ret.append(rgb_channels)
            else:
                rgba_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA)
                ret.append(rgba_image)
        return ret
    
if __name__ == "__main__":
    pass
