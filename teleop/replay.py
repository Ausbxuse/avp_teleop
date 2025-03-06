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
import struct

from robot_control.robot_hand import H1HandController
from teleop.robot_control.robot_arm import H1ArmController

FREQ = 30
DELAY = 1 / FREQ

from multiprocessing import Semaphore, shared_memory

import numpy as np

if __name__ == "__main__":
    merged_file_path = "./demos/default_task/20250305_202541/merged_data.jsonl"
    with open(merged_file_path, "r") as f:
        data_list = json.load(f)

    h1hand = H1HandController()
    h1arm = H1ArmController()

    q_poseList = np.zeros(35)
    q_tau_ff = np.zeros(35)
    right_angles = np.zeros(6)
    left_angles = np.zeros(6)

    last_time = None

    time.sleep(5)
    for i in range(len(data_list)):
        print("i:", i)
        q_poseList[13:27] = data_list[i]["arm_state"]
        right_angles = data_list[i]["hand_state"][0:6]
        left_angles = data_list[i]["hand_state"][6:12]

        if data_list[i]["ik_data"] is not None:
            q_tau_ff[13:27] = data_list[i]["ik_data"]["tau_ff"]
        if last_time is not None:
            pass_time = time.time() - last_time
            time.sleep(time_interval - pass_time)

        print("q_poseList", q_poseList)
        h1arm.SetMotorPose(q_poseList, q_tau_ff)
        # set hand
        h1hand.ctrl(right_angles, left_angles)

        if i != len(data_list) - 1:
            print(f"[DEBUG] {last_time}")
            time_interval = data_list[i + 1]["time"] - data_list[i]["time"]
            last_time = time.time()

    print("Replay Complete!")
