import atexit
import datetime
import json
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
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from Preprocessor import VuerPreprocessor
from TeleVision import OpenTeleVision

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pickle
import struct

from robot_control.robot_hand import H1HandController
from teleop.robot_control.robot_arm import H1ArmController
from teleop.robot_control.robot_arm_ik import Arm_IK

FREQ = 30
DELAY = 1 / FREQ

from multiprocessing import Semaphore, shared_memory

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
        self.semaphore = Semaphore(1)  # Ensures one process at a time

    def write_image(self, image):
        self.semaphore.acquire()
        try:
            np.copyto(self.img_array, image)
        finally:
            self.semaphore.release()

    def read_image(self):
        self.semaphore.acquire()
        try:
            image_copy = self.img_array.copy()
            return image_copy
        finally:
            self.semaphore.release()

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


class DataWriter:
    def __init__(self, dirname):
        # self.lock = threading.Lock()
        self.data = []
        self.filepath = os.path.join(dirname, "ik_data.json")

    def write_data(self, armtime, iktime, sol_q, tau_ff, head_rmat, left_pose, right_pose, index):
        # with self.lock:
        entry = {
            "index": index,
            "armtime": armtime,
            "iktime": iktime,
            "sol_q": sol_q.tolist(),
            "tau_ff": tau_ff.tolist(),
            "head_rmat": head_rmat.tolist(),
            "left_pose": left_pose.tolist(),
            "right_pose": right_pose.tolist(),
        }
        self.data.append(entry)

        with open(self.filepath, "w") as file:
            json.dump(self.data, file, indent=4)


# def motor_logger(dirname, stop_event, start_time, h1arm, h1hand):
#     log_filename = os.path.join(dirname, "motor_data.txt")
#     motor_count = 0
#     motor_data_list = []

#     try:
#         while not stop_event.is_set():
#             current_time = time.time() - start_time
#             if current_time >= DELAY * motor_count:
#                 armstate, armv = h1arm.GetMotorState()
#                 handstate = h1hand.get_hand_state()
#                 image_path = os.path.join(dirname, f"images/frame_{motor_count:06d}.jpg")
#                 motor_data = {
#                     "time": current_time,
#                     "arm_state": armstate.tolist(),
#                     "hand_state": handstate.tolist(),
#                     "image_path": image_path
#                 }
#                 motor_data_list.append(motor_data)
#                 motor_count += 1

#     except Exception as e:
#         print(f"[ERROR] motor_logger encountered an error: {e}")

#     finally:
#         with open(log_filename, "w") as f:
#             json.dump(motor_data_list, f, indent=4)

def image_receiver(image_queue, resolution, crop_size_w, crop_size_h):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.162:5555")

    while True:
        compressed_data = b""
        while True:
            chunk = socket.recv()
            compressed_data += chunk
            if len(chunk) < 60000:
                break
        data = zlib.decompress(compressed_data)
        frame_data = pickle.loads(data)

        # Decode and display the image
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        # print(frame.shape)
        sm.write_image(frame)
        # Control receiving frequency
        time.sleep(0.01)


def cleanup(lidar_proc):
    if lidar_proc.poll() is None:  # if the process is still running
        print("Sending SIGINT to the lidar_process...")
        lidar_proc.send_signal(signal.SIGINT)
        try:
            lidar_proc.wait(timeout=5)  # wait for it to terminate gracefully
        except lidar_proc.TimeoutExpired:
            lidar_proc.kill()  # force kill if needed


def rs_receiver(dirname, stop_event, start_time, h1arm, h1hand):
    #TODO: add depth of image

    """Receive and process frames from RS camera, then store filenames in the queue."""
    rs_filename = os.path.join(dirname, "rs.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rs_writer = None
    motor_data_list = []
    log_filename = os.path.join(dirname, "motor_data.txt")

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.162:5556")

    frame_count = 0
    next_capture_time = start_time
    print(f"[INFO] rs_receiver started. Saving video to {rs_filename}")

    try:
        while not stop_event.is_set():
            # print("#######################1#####################")
            compressed_data = b""
            while True:
                chunk = socket.recv()
                compressed_data += chunk
                if len(chunk) < 120000:  # Check for last chunk
                    break

            data = zlib.decompress(compressed_data)
            frame_data = pickle.loads(data)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            # print("#######################2#####################")
            resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            sm.write_image(resized_frame)



            if frame is None:
                print("[ERROR] Failed to decode frame!")
                continue  # Skip this frame if decoding failed
            # print("#######################3#####################")
            if rs_writer is None:
                frame_shape = frame.shape
                rs_writer = cv2.VideoWriter(
                    rs_filename, fourcc, 30, (frame_shape[1], frame_shape[0])
                )
            # print("#######################4#####################")
            current_time = time.time()
            if current_time >= next_capture_time:
                frame_filename = os.path.join(
                    dirname, f"images/frame_{frame_count:06d}.jpg"
                )
                cv2.imwrite(frame_filename, frame)
                # frame_queue.put(frame_filename)
                
                armstate, armv = h1arm.GetMotorState()
                handstate = h1hand.get_hand_state()
                imustate = h1arm.GetIMUState()
                motor_data = {
                    "index": frame_count,
                    "time": current_time,
                    "arm_state": armstate.tolist(),
                    "hand_state": handstate.tolist(),
                    "image_path": f"images/frame_{frame_count:06d}.jpg",
                    "imu_omega": imustate.omega,
                    "imu_rpy": imustate.rpy,
                    "ik_data": None,
                    "lidar": None   
                }
                motor_data_list.append(motor_data)
                frame_count += 1
                next_capture_time = start_time + frame_count * DELAY
            # print("#######################5#####################")
            rs_writer.write(frame)
            time.sleep(0.05)

    except KeyboardInterrupt as e:
        print(f"[INTR] keyboard interrupted: {e}")
    except Exception as e:
        print(f"[ERROR] rs_receiver encountered an error: {e}")

    finally:
        if rs_writer:
            rs_writer.release()
            print(f"[INFO] Video saved successfully as {rs_filename}")
        with open(log_filename, "w") as f:
            json.dump(motor_data_list, f, indent=4)
        context.term()
        print("[INFO] rs_receiver process has exited.")

def ik_is_ready(ik_data_list, time_key):
    closest_ik_entry = min(ik_data_list, key=lambda x: abs(x["armtime"] - time_key))
    if abs(closest_ik_entry["armtime"] - time_key) > DELAY/2:
        return False, None
    # print("closest_ik_entry found", closest_ik_entry["armtime"], time_key)
    return True, closest_ik_entry

def lidar_is_ready(lidar_time_list, time_key):
    closest_lidar_entry = min(lidar_time_list, key=lambda x: abs(x - time_key))
    if abs(closest_lidar_entry - time_key) > DELAY/2:
        return False, None
    # print("closest_ik_entry found", closest_ik_entry["armtime"], time_key)
    return True, closest_lidar_entry


def merge_data_to_pkl(motor_data_path, ik_data_path, lidar_data_path, output_path):
    lidar_time_list = []


    lidar_files = [f for f in os.listdir(lidar_data_path) if os.path.isfile(os.path.join(lidar_data_path, f))]

    for lidar_file_name in lidar_files:
        time_parts = lidar_file_name.split(".")[0:2]
        lidar_time_list.append(float(time_parts[0] + "." + time_parts[1]))
    

    print("loading")
    with open(motor_data_path, "r") as f:
        robot_data_json_list = json.load(f)

    with open(ik_data_path, "r") as f:
        ik_data_list = json.load(f)

    ik_data_dict = {entry["armtime"]: entry for entry in ik_data_list}
    robot_data_dict = {entry["time"]: entry for entry in robot_data_json_list}

    last_motor_data = None

    for motor_entry in robot_data_json_list:
        time_key = motor_entry["time"]
        ik_ready_flag, closest_ik_entry  = ik_is_ready(ik_data_list, time_key)
        if ik_ready_flag:
            robot_data_dict[time_key]["ik_data"] = ik_data_dict[closest_ik_entry["armtime"]]
            last_motor_data = robot_data_dict[time_key]["ik_data"]
        else:
            robot_data_dict[time_key]["ik_data"] = last_motor_data 

        # merge lidar path
        lidar_ready_flag, closest_lidar_time = lidar_is_ready(lidar_time_list, time_key)
        if lidar_ready_flag:
            robot_data_dict[time_key]["lidar"] = os.path.join("lidar", f"{closest_lidar_time}.pcd")


    # with open(output_path, "wb") as f:
    #     print("4")
    #     pickle.dump(motor_data_list, f)
    with open(output_path, "w") as f:
            json.dump(robot_data_json_list, f, indent=4)
    
    print(f"Mergefile saved to {output_path}")

    




def profile(name):
    with open(f"profile.txt", "a") as file:
        time = datetime.datetime.now()
        file.write(f"{time}:{name}\n")


if __name__ == "__main__":

    manager = Manager()
    #stop_event = threading.Event()
    stop_event =Event()

    teleoperator = VuerTeleop("inspire_hand.yml")
    h1hand = H1HandController()
    h1arm = H1ArmController()
    arm_ik = Arm_IK()
    sm = SharedMemoryImage((720, 640))
    global rs_thread
    global motor_thread
    global dirname
    image_process = Process(target=image_receiver, args=(sm, teleoperator.resolution, teleoperator.crop_size_w, teleoperator.crop_size_h))
    

    try:
        user_input = input(
            "Please enter the start signal (enter 's' to start the subsequent program): "
        )
        if user_input.lower() == "s":
            #image_process.start()
            loop_idx = 0

            dirname = time.strftime("demo_%Y%m%d_%H%M%S")
            start_time = time.time()
            proc = subprocess.Popen(["./point_cloud_recorder", "./mid360_config.json", dirname + "/lidar"])
            os.mkdir(dirname)

            images_dir = os.path.join(dirname, "images")
            os.mkdir(images_dir)
            data_writer = DataWriter(dirname)

            

            rs_thread = Process(
                target=rs_receiver,
                args=(dirname, stop_event, start_time, h1arm, h1hand),
            )
            rs_thread.start()
            # import pdb; spdb.set_trace()
            # motor_thread = threading.Thread(
            #     target=mot    or_logger, args=(dirname, stop_event, start_time, h1arm, h1hand)
            # )
            # motor_thread.start()

            while not stop_event.is_set():
                # profile("Main loop started")
                # time.sleep(0.05)
                frame = sm.read_image()
                # print("#hehehehehhe")
                # print("frame shape", frame.shape)
                np.copyto(teleoperator.img_array, np.array(frame))

                # print("#hehehehehh 222222")

                #profile("get imu finished")
                armstate, armv = h1arm.GetMotorState()
                # profile("get arm finished")
                handstate = h1hand.get_hand_state()
                # print("#hehehehehh 33333")

                # profile("get hand finished")
                motor_time = time.time()

                head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                    teleoperator.step()
                )
                #TODO : add hand pose ik
                # print("#hehehehehh 444444")
                # profile("ik teleop finished")
                sol_q, tau_ff, flag = arm_ik.ik_fun(
                    left_pose, right_pose, armstate, armv
                )
                # print("################ik:", sol_q)
                ik_time = time.time()
                # t = datetime.datetime.now()

                # profile("ik finished")

                data_writer.write_data(motor_time, ik_time, sol_q, tau_ff, head_rmat, left_pose, right_pose, loop_idx)
                loop_idx += 1

                q_poseList = np.zeros(35)
                q_tau_ff = np.zeros(35)

                # if flag:
                #     q_poseList[13:27] = sol_q
                #     q_tau_ff[13:27] = tau_ff
                # else:
                #     q_poseList[13:27] = armstate
                #     q_tau_ff = np.zeros(35)

                if right_qpos is not None and left_qpos is not None:
                    right_angles = [1.7 - right_qpos[i] for i in [4, 6, 2, 0]]
                    right_angles.append(1.2 - right_qpos[8])
                    right_angles.append(0.5 - right_qpos[9])
                
                    left_angles = [1.7 - left_qpos[i] for i in [4, 6, 2, 0]]
                    left_angles.append(1.2 - left_qpos[8])
                    left_angles.append(0.5 - left_qpos[9])
                    h1hand.crtl(right_angles, left_angles)

    except KeyboardInterrupt:
        print("Recording ended!")

        stop_event.set()
        rs_thread.join(timeout=1)
        

    ik_filepath = os.path.join(dirname, "ik_data.json")
    motor_filepath = os.path.join(dirname, "motor_data.txt")
    output_filepath = os.path.join(dirname, "merged_data.json")
    lidar_filepath = os.path.join(dirname, "lidar")
    merge_data_to_pkl(motor_filepath, ik_filepath, lidar_filepath, output_filepath)

