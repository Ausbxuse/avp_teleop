import atexit
import datetime
import json
import math
import os
import pickle
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
