import datetime
import json
import os
import pickle
import sys
import threading
import time
import zlib
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory
from pathlib import Path

import cv2
import numpy as np
import yaml
import zmq
from dex_retargeting.retargeting_config import RetargetingConfig

from constants_vuer import tip_indices
from Preprocessor import VuerPreprocessor
from TeleVision import OpenTeleVision

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import struct

from robot_control.robot_hand import H1HandController
from teleop.robot_control.robot_arm import H1ArmController
from teleop.robot_control.robot_arm_ik import Arm_IK


class VuerTeleop:
    def __init__(self, config_file_path):
        # self.resolution = (720,1280) #(480,640)
        self.resolution = (720, 640)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w,
        )

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(
            create=True, size=np.prod(self.img_shape) * np.uint8().itemsize
        )
        self.img_array = np.ndarray(
            (self.img_shape[0], self.img_shape[1], 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(
            self.resolution_cropped, self.shm.name, image_queue, toggle_streaming
        )
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir("../assets")
        with Path(config_file_path).open("r") as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg["left"])
        right_retargeting_config = RetargetingConfig.from_dict(cfg["right"])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = (
            self.processor.process(self.tv)
        )
        head_rmat = head_mat[:3, :3]

        left_wrist_mat[2, 3] += 0.55
        right_wrist_mat[2, 3] += 0.55
        left_wrist_mat[0, 3] += 0.05
        right_wrist_mat[0, 3] += 0.05

        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[
            [4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]
        ]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[
            [4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]
        ]

        return head_rmat, left_wrist_mat, right_wrist_mat, left_qpos, right_qpos


class DataWriter:
    def __init__(self, dirname):
        self.lock = threading.Lock()
        self.data = []
        self.filepath = os.path.join(dirname, "data_log.json")

    def write_data(self, timestamp, image_filename, arm_state, hand_state):
        with self.lock:
            entry = {
                "timestamp": timestamp.isoformat(),
                "arm_state": arm_state.tolist(),
                "hand_state": hand_state.tolist(),
                "image_path": image_filename,
            }
            self.data.append(entry)

            with open(self.filepath, "w") as file:
                json.dump(self.data, file, indent=4)


def rs_receiver(dirname, frame_queue, stop_event):
    """Receive and process frames from RS camera, then store filenames in the queue."""
    rs_filename = os.path.join(dirname, "rs.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rs_writer = None

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.162:5556")

    frame_count = 0
    print(f"[INFO] rs_receiver started. Saving video to {rs_filename}")

    try:
        while not stop_event.is_set():
            compressed_data = b""
            while True:
                chunk = socket.recv()
                compressed_data += chunk
                if len(chunk) < 120000:  # Check for last chunk
                    break

            data = zlib.decompress(compressed_data)
            frame_data = pickle.loads(data)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

            if frame is None:
                print("[ERROR] Failed to decode frame!")
                continue  # Skip this frame if decoding failed

            if rs_writer is None:
                frame_shape = frame.shape
                rs_writer = cv2.VideoWriter(
                    rs_filename, fourcc, 30, (frame_shape[1], frame_shape[0])
                )

            frame_filename = os.path.join(dirname, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)

            rs_writer.write(frame)
            frame_queue.put(frame_filename)

            frame_count += 1

    except Exception as e:
        print(f"[ERROR] rs_receiver encountered an error: {e}")

    finally:
        if rs_writer:
            rs_writer.release()
            print(f"[INFO] Video saved successfully as {rs_filename}")
        context.term()
        print("[INFO] rs_receiver process has exited.")


def profile(name):
    with open(f"profile.txt", "a") as file:
        time = datetime.datetime.now()
        file.write(f"{time}:{name}\n")


if __name__ == "__main__":
    manager = Manager()
    frame_queue = manager.Queue()
    stop_event = threading.Event()

    teleoperator = VuerTeleop("inspire_hand.yml")
    h1hand = H1HandController()
    h1arm = H1ArmController()
    arm_ik = Arm_IK()

    try:
        user_input = input(
            "Please enter the start signal (enter 's' to start the subsequent program): "
        )
        if user_input.lower() == "s":
            dirname = time.strftime("demo_%Y%m%d_%H%M%S")
            os.mkdir(dirname)
            data_writer = DataWriter(dirname)

            rs_thread = threading.Thread(
                target=rs_receiver, args=(dirname, frame_queue, stop_event)
            )
            rs_thread.start()

            while True:
                profile("Main loop started")
                # time.sleep(0.05)

                frame_filename = (
                    frame_queue.get()
                    if not frame_queue.empty()
                    else "frame_not_available.jpg"
                )

                armstate, armv = h1arm.GetMotorState()
                profile("get arm finished")
                handstate = h1hand.get_hand_state()
                profile("get hand finished")
                head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                    teleoperator.step()
                )
                profile("ik teleop finished")
                sol_q, tau_ff, flag = arm_ik.ik_fun(
                    left_pose, right_pose, armstate, armv
                )
                t = datetime.datetime.now()

                profile("ik finished")

                data_writer.write_data(t, frame_filename, armstate, handstate)
                print("write data", datetime.datetime.now())

                q_poseList = np.zeros(35)
                q_tau_ff = np.zeros(35)

                # if flag:
                #     q_poseList[13:27] = sol_q
                #     q_tau_ff[13:27] = tau_ff
                # else:
                #     q_poseList[13:27] = armstate
                #     q_tau_ff = np.zeros(35)

                # if right_qpos is not None and left_qpos is not None:
                #     right_angles = [1.7 - right_qpos[i] for i in [4, 6, 2, 0]]
                #     right_angles.append(1.2 - right_qpos[8])
                #     right_angles.append(0.5 - right_qpos[9])
                #
                #     left_angles = [1.7 - left_qpos[i] for i in [4, 6, 2, 0]]
                #     left_angles.append(1.2 - left_qpos[8])
                #     left_angles.append(0.5 - left_qpos[9])
                #     h1hand.crtl(right_angles, left_angles)

    except KeyboardInterrupt:
        print("Recording ended!")
    finally:
        stop_event.set()
        rs_thread.join()
        exit(0)
