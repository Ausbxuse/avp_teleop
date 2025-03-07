import argparse
import json
import logging
import os
import pickle
import queue
import signal
import subprocess
import sys
import threading
import time
import zlib
from multiprocessing import Event, Lock, Process, Queue, shared_memory

import cv2
import numpy as np
import zmq

from robot_control.robot_arm import H1ArmController
from robot_control.robot_arm_ik import Arm_IK
from robot_control.robot_hand import H1HandController
from utilities import VuerTeleop

# --------------------- Debug Logger Setup ---------------------
logger = logging.getLogger("robot_teleop")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# --------------------------------------------------------------
import numpy as np

from utilities import VuerTeleop

teleoperator = VuerTeleop("inspire_hand.yml")

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://192.168.123.162:5556")
socket.setsockopt(zmq.RCVTIMEO, 200)
socket.setsockopt(zmq.RCVHWM, 1)


while True:
    compressed_data = b""
    while True:  # TODO: verify correctness
        chunk = socket.recv()
        compressed_data += chunk
        if len(chunk) < 120000:  # Check for last chunk
            break

    try:
        data = zlib.decompress(compressed_data)
        frame_data = pickle.loads(data)
    except Exception as e:
        logger.error(f"Failed decompressing or unpickling frame data: {e}")

    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)  # np: (height, width)

    if frame is None:
        logger.error("Failed to decode frame!")

    color_frame = frame[:, : frame.shape[1] // 2]
    depth_frame = frame[:, frame.shape[1] // 2 :]

    if color_frame is not None:
        resized_frame = cv2.resize(
            color_frame, (1280, 720), interpolation=cv2.INTER_LINEAR
        )
    np.copyto(teleoperator.img_array, resized_frame)
