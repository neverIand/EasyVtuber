from multiprocessing import Process, Value, shared_memory
from .args import args
import math
import socket
import numpy as np
from .utils.shared_mem_guard import SharedMemoryGuard
from .utils.fps import FPS
from .utils.filter import OneEuroFilterNumpy
from .utils.open_see_face import parse_open_see_face_packet
from OneEuroFilter import OneEuroFilter
import time

class OSFClientProcess(Process):
    def __init__(self, pose_position_shm: shared_memory.SharedMemory):
        super().__init__()
        self.pose_position_shm = pose_position_shm
        self.address = args.osf_input.split(':')[0]
        self.port = int(args.osf_input.split(':')[1])
        self.fps = Value('f', 0.0)

    def run(self):
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("", self.port))
        self.socket.settimeout(None) # wait indefinitely

        print("Warming up OpenSeeFace connection...")

        frame_count = 0
        input_fps = FPS(60)
        while frame_count < 5:
            socket_bytes = self.socket.recv(8192)
            if not socket_bytes:
                raise Exception("Can't receive OpenSeeFace data (stream end?).")
            input_fps()
            frame_count += 1

        position_vector_0 = None
        position_vector = np.array([0, 0, 0, 1], dtype=np.float32)
        model_input_arr = np.zeros(45, dtype=np.float32)
        # 呼吸循环参数
        breath_start_time = time.perf_counter()
        pose_filter = OneEuroFilterNumpy(freq=input_fps.view(), mincutoff=args.filter_min_cutoff, beta=args.filter_beta)
        position_filter = OneEuroFilterNumpy(freq=input_fps.view(), mincutoff=args.filter_min_cutoff, beta=args.filter_beta)
        iris_x_filter = OneEuroFilter(freq=input_fps.view(), mincutoff=0.001, beta=1.0) # extra filter for iris movement
        iris_y_filter = OneEuroFilter(freq=input_fps.view(), mincutoff=0.001, beta=1.0) # extra filter for iris movement
        rotation_offset = None
        print("OpenSeeFace Input Running at {:.2f} FPS".format(input_fps.view()))
        while True:
            socket_bytes = self.socket.recv(8192)

            if not socket_bytes:
                raise Exception("Can't receive OpenSeeFace data (stream end?).")
            
            self.fps.value = input_fps()

            try:
                data = parse_open_see_face_packet(socket_bytes)
            except Exception:
                print("OpenSeeFace data parse error:", socket_bytes)
                continue

            # 计算呼吸效果（使用 sin 函数，在 breath_cycle 时间内从 0 到 1 再到 0）
            if math.isfinite(args.breath_cycle):
                breath_elapsed = (time.perf_counter() - breath_start_time) % args.breath_cycle
                # 使用 sin 函数，让值在一个周期内从 0 -> 1 -> 0
                # sin 在 0 到 π 之间从 0 到 1 到 0
                breath_value = np.sin(breath_elapsed / args.breath_cycle * np.pi)
            else:
                breath_value = 0.0

            model_input_arr.fill(0.0)
            model_input_arr[14] = 1 - data.left_eye_open
            model_input_arr[15] = 1 - data.right_eye_open
            model_input_arr[26] = max(data.mouth_open, 0) * 2 # Open larger mouth
            model_input_arr[37] = iris_x_filter(
                -data.eye_rotation_y * 3 - data.rotation_x / 57.3 * 1.5,
                timestamp=time.perf_counter(),
            )
            model_input_arr[38] = iris_y_filter(
                data.eye_rotation_x * 3 + data.rotation_y / 57.3,
                timestamp=time.perf_counter(),
            )
            model_input_arr[6] = data.eyebrow_left
            model_input_arr[7] = data.eyebrow_right
            if rotation_offset is None:
                rotation_offset = [data.rotation_x, data.rotation_y, data.rotation_z]
            model_input_arr[39] = (data.rotation_x - rotation_offset[0]) / 57.3 * 3
            model_input_arr[40] = -(data.rotation_y - rotation_offset[1]) / 57.3 * 3
            model_input_arr[41] = (data.rotation_z - rotation_offset[2]) / 57.3 * 2
            model_input_arr[42] = model_input_arr[40]
            model_input_arr[43] = model_input_arr[41]
            model_input_arr[44] = breath_value

            if position_vector_0 is None: #Provide an initial reference point
                position_vector_0 = [data.translation_x, data.translation_y, data.translation_z]
            #Compute relative translation
            position_vector[0] = -(data.translation_x - position_vector_0[0]) * 0.1
            position_vector[1] = -(data.translation_y - position_vector_0[1]) * 0.1
            position_vector[2] = -(data.translation_z - position_vector_0[2]) * 0.1

            with pose_position_shm_guard.lock():
                np_pose_shm[:] = pose_filter(model_input_arr)
                np_position_shm[:] = position_filter(position_vector)
                # np_pose_shm[:] = np.array(model_input_arr, dtype=np.float32)
                # np_position_shm[:] = np.array(position_vector, dtype=np.float32)
