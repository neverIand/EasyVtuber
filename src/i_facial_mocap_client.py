import math
from multiprocessing import Process, shared_memory, Value
import socket
import time

import numpy as np

from tha2.mocap.ifacialmocap_constants import *
from .args import args
from .utils.fps import FPS
from .utils.ifacialmocap import (
    fill_ifacialmocap_model_input,
    parse_ifacialmocap_v1_pose,
)
from .utils.shared_mem_guard import SharedMemoryGuard


class IFMClientProcess(Process):
    def __init__(self, pose_position_shm: shared_memory.SharedMemory):
        super().__init__()
        self.pose_position_shm = pose_position_shm
        self.address = args.ifm_input.split(':')[0]
        self.port = int(args.ifm_input.split(':')[1])
        self.fps = Value('f', 0.0)
    def run(self):
        # Import and construct the SciPy-based converter only in the input
        # subprocess.  On Windows this keeps the parent/launcher startup path
        # lightweight while preserving the exact legacy conversion routine.
        from tha2.poser.modes.mode_20_wx import IFacialMocapPoseConverter20

        ifm_converter = IFacialMocapPoseConverter20()
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        
        handshake_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            data = "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719".encode('utf-8')
            handshake_socket.sendto(data, (self.address, self.port))
        finally:
            handshake_socket.close()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("", self.port))
        self.socket.settimeout(5.0) # give ifm app some time to respond

        print("Warming up iFacialMocap connection...")

        frame_count = 0
        input_fps = FPS(60)
        while frame_count < 60:
            socket_bytes = self.socket.recv(8192)
            if not socket_bytes:
                raise Exception("Can't receive iFacialMocap data (stream end?).")
            input_fps()
            frame_count += 1

        self.socket.settimeout(0.5) # 500 ms

        # 呼吸循环参数
        breath_start_time = time.perf_counter()
        print("iFacialMocap Input Running at {:.2f} FPS".format(input_fps.view()))
        model_input_arr = np.empty(45, dtype=np.float32)
        position_vector = np.empty(4, dtype=np.float32)
        last_parse_error_time = float('-inf')
        while True:
            try:
                socket_bytes = self.socket.recv(8192)
            except socket.timeout:
                continue

            if not socket_bytes:
                raise Exception("Can't receive iFacialMocap data (stream end?).")
            
            self.fps.value = input_fps()

            try:
                socket_string = socket_bytes.decode("utf-8")
                data = parse_ifacialmocap_v1_pose(socket_string)
                ifacialmocap_pose_converted = ifm_converter.convert(data)
            except Exception as error:
                now = time.perf_counter()
                if now - last_parse_error_time >= 5.0:
                    print("iFacialMocap data parse error:", error)
                    last_parse_error_time = now
                continue

            # 计算呼吸效果（使用 sin 函数，在 breath_cycle 时间内从 0 到 1 再到 0）
            if math.isfinite(args.breath_cycle):
                breath_elapsed = (time.perf_counter() - breath_start_time) % args.breath_cycle
                breath_value = np.sin(breath_elapsed / args.breath_cycle * np.pi)
            else:
                breath_value = 0.0

            fill_ifacialmocap_model_input(
                ifacialmocap_pose_converted,
                breath_value,
                model_input_arr,
            )
            position_vector[:] = data[HEAD_BONE_QUAT]

            with pose_position_shm_guard.lock():
                # np_pose_shm[:] = pose_filter(np.array(model_input_arr, dtype=np.float32))
                # np_position_shm[:] = position_filter(np.array(position_vector, dtype=np.float32))
                np_pose_shm[:] = model_input_arr
                np_position_shm[:] = position_vector

    @staticmethod
    def convert_from_blender_data(blender_data):
        return parse_ifacialmocap_v1_pose(blender_data)
