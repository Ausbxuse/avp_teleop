import atexit
import datetime
import json
import math
import os
import queue
import signal
import subprocess
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
import pickle

FREQ = 30
DELAY = 1 / FREQ

import numpy as np


class AsyncImageWriter:
    def __init__(self):
        self.queue = queue.Queue()
        self.kill_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.kill_event.is_set() or not self.queue.empty():
            try:
                filename, image = self.queue.get(timeout=0.5)
                cv2.imwrite(filename, image)
            except queue.Empty:
                continue

    def write_image(self, filename, image):
        self.queue.put((filename, image))

    def close(self):
        self.kill_event.set()
        self.thread.join()

class AsyncWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.queue = queue.Queue()
        self.kill_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        with open(self.filepath, "a") as f:
            while not self.kill_event.is_set() or not self.queue.empty():
                try:
                    item = self.queue.get(timeout=0.5)
                    # logger.debug(f"async writer: writing elements {item}")
                    f.write(item + "\n")
                    # f.flush()
                except queue.Empty:
                    continue

    def write(self, item):
        self.queue.put(item)

    def close(self):
        self.kill_event.set()
        self.thread.join()

class IKDataWriter:
    def __init__(self, dirname):
        self.buffer = []
        self.filepath = os.path.join(dirname, "ik_data.jsonl")
        self.async_writer = AsyncWriter(os.path.join(dirname, "ik_data.jsonl"))

    def write_data(
        self,
        # right_angles,
        # left_angles,
        arm_time,
        ik_time,
        sol_q,
        tau_ff,
        head_rmat,
        left_pose,
        right_pose,
    ):
        entry = {
            # "right_angles": right_angles,
            # "left_angles": left_angles,
            "armtime": arm_time,
            "iktime": ik_time,
            "sol_q": sol_q.tolist(),
            "tau_ff": tau_ff.tolist(),
            "head_rmat": head_rmat.tolist(),
            "left_pose": left_pose.tolist(),
            "right_pose": right_pose.tolist(),
        }
        self.async_writer.write(json.dumps(entry))
    def close(self):
        self.async_writer.close()


class DataMerger:
    def __init__(self, dirname) -> None:
        self.robot_data_path = os.path.join(dirname, "robot_data.jsonl")
        self.ik_data_path = os.path.join(dirname, "ik_data.jsonl")
        self.lidar_data_path = os.path.join(dirname, "lidar")
        self.output_path = os.path.join(dirname, "merged_data.jsonl")

    def _ik_is_ready(self, ik_data_list, time_key):
        closest_ik_entry = min(ik_data_list, key=lambda x: abs(x["armtime"] - time_key))
        if abs(closest_ik_entry["armtime"] - time_key) > DELAY / 2:
            return False, None
        return True, closest_ik_entry

    def _lidar_is_ready(self, lidar_time_list, time_key):
        closest_lidar_entry = min(lidar_time_list, key=lambda x: abs(x - time_key))
        if abs(closest_lidar_entry - time_key) > DELAY / 2:
            return False, None
        return True, closest_lidar_entry

    def merge_json(self):  # TODO: merge to pkl
        lidar_time_list = []

        lidar_files = [
            f
            for f in os.listdir(self.lidar_data_path)
            if os.path.isfile(os.path.join(self.lidar_data_path, f))
        ]

        for lidar_file_name in lidar_files:
            time_parts = lidar_file_name.split(".")[0:2]
            lidar_time_list.append(float(time_parts[0] + "." + time_parts[1]))

        logger.info("loading robot and IK data for merging.")
        robot_data_json_list = []
        with open(self.robot_data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    robot_data_json_list.append(json.loads(line))

        ik_data_list = []
        with open(self.ik_data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    ik_data_list.append(json.loads(line))

        ik_data_dict = {entry["armtime"]: entry for entry in ik_data_list}
        robot_data_dict = {entry["time"]: entry for entry in robot_data_json_list}

        if ik_data_list[0]["armtime"] > robot_data_json_list[0]["time"]:
            last_robot_data = None
        else:
            last_robot_data = ik_data_list[0]

        for motor_entry in robot_data_json_list:
            time_key = motor_entry["time"]
            ik_ready_flag, closest_ik_entry = self._ik_is_ready(ik_data_list, time_key)
            if ik_ready_flag and closest_ik_entry is not None:
                robot_data_dict[time_key]["ik_data"] = ik_data_dict[
                    closest_ik_entry["armtime"]
                ]
                last_robot_data = robot_data_dict[time_key]["ik_data"]
            else:
                robot_data_dict[time_key]["ik_data"] = last_robot_data

            # merge lidar path
            lidar_ready_flag, closest_lidar_time = self._lidar_is_ready(
                lidar_time_list, time_key
            )
            if lidar_ready_flag:
                robot_data_dict[time_key]["lidar"] = os.path.join(
                    "lidar", f"{closest_lidar_time}.pcd"
                )

        with open(self.output_path, "w") as f:
            json.dump(robot_data_json_list, f, indent=4)

        logger.info(f"Mergefile saved to {self.output_path}")




class SharedMemoryImage:
    def __init__(self, img_shape):
        self.resolution = img_shape  # (720, 1280)
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
        self.lock = Lock()  # Ensures one process at a time

    def write_image(self, image):
        with self.lock:
            np.copyto(self.img_array, image)

    def read_image(self):
        with self.lock:
            image_copy = self.img_array.copy()
            return image_copy

    def cleanup(self):
        self.shm.close()
        self.shm.unlink()


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

    def shutdown(self):
        self.shm.close()
        self.shm.unlink()
        self.tv.shutdown()
