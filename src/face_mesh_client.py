import mediapipe as mp
from multiprocessing import Process, shared_memory, Value
import cv2
import math
import numpy as np
from .args import args
from .utils.shared_mem_guard import SharedMemoryGuard
from .utils.camera_frame import prepare_mediapipe_rgb
from .utils.pose import get_pose
from .utils.fps import FPS
from .utils.filter import OneEuroFilterNumpy
from OneEuroFilter import OneEuroFilter
import time


class FaceMeshClientProcess(Process):
    def __init__(self, pose_position_shm: shared_memory.SharedMemory):
        super().__init__()
        self.pose_position_shm = pose_position_shm
        self.fps = Value('f', 0.0)

    def run(self):
        facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("Can't open webcam")
        
        print("Warming up webcam...")

        frame_count = 0
        input_fps = FPS(60)
        while frame_count < 5:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Can't receive frame (stream end?).")
            input_fps()
            frame_count += 1

        # 呼吸循环参数
        breath_start_time = time.perf_counter()
        pose_filter = OneEuroFilterNumpy(freq=input_fps.view(), mincutoff=args.filter_min_cutoff, beta=args.filter_beta)
        position_offset = None
        print("Webcam Input Running at {:.2f} FPS".format(input_fps.view()))
        position_vector = np.array([0, 0, 0, 1], dtype=np.float32)
        model_input_arr = np.zeros(45, dtype=np.float32)
        rgb_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Can't receive frame (stream end?).")
            self.fps.value = input_fps()
            # Reuse the RGB frame allocation; MediaPipe processes it
            # synchronously before the next camera frame overwrites it.
            rgb_frame = prepare_mediapipe_rgb(frame, rgb_frame)
            results = facemesh.process(rgb_frame)
            if results.multi_face_landmarks is None:
                continue

            if math.isfinite(args.breath_cycle):
                # 使用 sin 函数，让值在一个周期内从 0 -> 1 -> 0。
                breath_elapsed = (time.perf_counter() - breath_start_time) % args.breath_cycle
                breath_value = np.sin(breath_elapsed / args.breath_cycle * np.pi)
            else:
                breath_value = 0.0

            facial_landmarks = results.multi_face_landmarks[0].landmark
            pose = get_pose(facial_landmarks)
            eye_l_h_temp = pose[0]
            eye_r_h_temp = pose[1]
            mouth_ratio = pose[2]
            eye_y_ratio = pose[3]
            eye_x_ratio = pose[4]
            x_angle = pose[5]
            y_angle = pose[6]
            z_angle = pose[7]

            if position_offset is None:
                position_offset = [(x_angle - 1.5) * 1.6, y_angle * 2.0 , (z_angle + 1.5) * 2]

            # Reuse the model input buffer instead of allocating three lists and
            # a new NumPy array for every camera frame.
            model_input_arr.fill(0.0)
            blink = max(eye_l_h_temp, eye_r_h_temp)
            model_input_arr[14] = blink
            model_input_arr[15] = blink
            model_input_arr[26] = mouth_ratio * 2.0
            model_input_arr[39] = (x_angle - 1.5) * 1.6 - position_offset[0]
            model_input_arr[40] = y_angle * 2.0 - position_offset[1]
            model_input_arr[41] = (z_angle + 1.5) * 2 - position_offset[2]
            model_input_arr[42] = model_input_arr[40]
            model_input_arr[43] = model_input_arr[41]
            model_input_arr[44] = breath_value

            with pose_position_shm_guard.lock():
                np_pose_shm[:] = pose_filter(model_input_arr)
                np_position_shm[:] = position_vector
