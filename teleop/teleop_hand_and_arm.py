import datetime
import os
import pickle
import sys
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


def rs_receiver(dirname):
    # Create video writer inside this function
    rs_filename = time.strftime(f"{dirname}/rs.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Open a dummy frame first to determine dimensions
    frame_shape = (480, 640, 3)  # Default shape if first frame fails
    rs_writer = None

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.162:5556")

    print(f"[INFO] rs_receiver started. Saving video to {rs_filename}")

    try:
        while True:
            compressed_data = b""
            while True:
                chunk = socket.recv()
                compressed_data += chunk
                if len(chunk) < 120000:  # Check for last chunk
                    break
            
            # Decompress and decode frame
            data = zlib.decompress(compressed_data)
            frame_data = pickle.loads(data)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

            if frame is None:
                print("[ERROR] Failed to decode frame!")
                continue  # Skip this frame if decoding failed
            
            # Initialize video writer only when the first frame is received
            if rs_writer is None:
                frame_shape = frame.shape
                print(f"[INFO] Frame shape: {frame_shape}")
                rs_writer = cv2.VideoWriter(rs_filename, fourcc, 30, (frame_shape[1], frame_shape[0]))

            rs_writer.write(frame)  # Write frame to video

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received, stopping rs_receiver...")
        sys.exit(0)  # Properly exit the process

    except Exception as e:
        print(f"[ERROR] rs_receiver encountered an error: {e}")
        sys.exit(1)  # Ensure the process exits with failure code

    finally:
        if rs_writer:
            rs_writer.release()
            print(f"[INFO] Video saved successfully as {rs_filename}")
        context.term()
        print("[INFO] rs_receiver process has exited.")


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
        self.lock = Lock()

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


if __name__ == "__main__":
    manager = Manager()
    image_queue = manager.Queue()
    teleoperator = VuerTeleop("inspire_hand.yml")
    h1hand = H1HandController()
    h1arm = H1ArmController()
    arm_ik = Arm_IK()
    # sm = SharedMemoryImage((720,1280))
    # sm = SharedMemoryImage((480,640))

    # TODO: overlapping videos in vision pro
    sm = SharedMemoryImage((720, 640))
    # image_process = Process(target=image_receiver, args=(sm, teleoperator.resolution, teleoperator.crop_size_w, teleoperator.crop_size_h))
    # image_process.start()

    try:
        user_input = input(
            "Please enter the start signal (enter 's' to start the subsequent program):"
        )
        if user_input.lower() == "s":
            dirname = time.strftime("demo_%Y%m%d_%H%M%S")
            os.mkdir(dirname)
            q_poseList = np.zeros(35)
            is_first = True
            rs_filename = time.strftime(f"{dirname}/rs.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # video_writer = None
            # rs_writer = cv2.VideoWriter(rs_filename, fourcc, 30, (480, 1280))
            rs_process = Process(target=rs_receiver, args=(dirname,))
            rs_process.start()
            while True:

                armstate = None
                armv = None
                frame = sm.read_image()

                np.copyto(teleoperator.img_array, np.array(frame))
                # print(frame.shape)

                # # begin recording
                # if video_writer is None:
                #     video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (frame.shape[1], frame.shape[0]))
                #
                # video_writer.write(np.array(frame))

                # rs_writer.write(np.array(rs_frame))

                handstate = h1hand.get_hand_state()

                q_tau_ff = np.zeros(35)
                armstate, armv = h1arm.GetMotorState()
                head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                    teleoperator.step()
                )

                sol_q, tau_ff, flag = arm_ik.ik_fun(
                    left_pose, right_pose, armstate, armv
                )
                t = datetime.datetime.now()

                with open(f"{dirname}/arm.txt", "a") as file:
                    file.write(f"[{t}]: {armstate}\n")
                with open(f"{dirname}/hand.txt", "a") as file:
                    file.write(f"[{t}]: {handstate}\n")

                # print("Sol_q[0:7]:", sol_q[0:7])
                # print("Sol_q[7:14]", sol_q[7:14])

                # with open("sol.txt", "a") as file:
                #     file.write(f"Sol_q: {sol_q}\n")
                #     # file.write(f"Head_rmat: {head_rmat}\n")
                #     file.write(f"Left_Pose: {left_pose}\n")
                #     file.write(f"Right_pose: {right_pose}\n")

                # print("Tau_ff:", tau_ff)
                # print("Flag:", flag)

                for idx, q in enumerate(sol_q):
                    if q >= 0:
                        q = q % 3.14
                        if q >= 2.3:
                            q = q - 3.14
                    else:
                        q = q % 3.14 - 3.14
                        if q <= -2.3:
                            q = q + 3.14
                    sol_q[idx] = q
                if sol_q[0] > 1.4:
                    sol_q[0] -= 3.14

                # with open("sol.txt", "a") as file:
                #     file.write(f"Sol_q after: {sol_q}\n\n\n\n")

                # #             if np.all((sol_q % 3.14 > 3.14 - 1) | (sol_q % 3.14 < 1)):
                # #                 print("solution found: ", i, j, k)

                # print("Tau_ff:", tau_ff)
                # print("Flag:", flag)

                #                 with open("test.txt", 'w') as file:
                #                     file.write(f"Sol_q: {sol_q}\n")
                #                     file.write(f"{i},{j},{k}\n")
                # break

                # sol_q = np.ones(14) * 0.2
                # for idx, target in enumerate(sol_q):
                #     print("rotating index:", idx+13)
                #     last = q_poseList[idx+13]
                #     if target <= 0:
                #         for i in range((last*100).astype(np.int32), (last * 100+ target * 100).astype(np.int32),-1):
                #             time.sleep(0.04)
                #             print("setting target to", i/100.0)
                #             q_poseList[idx+13] = i/100.0
                #             h1arm.SetMotorPose(q_poseList, q_tau_ff)
                #     else:
                #         for i in range((last*100).astype(np.int32), (last * 100+ target * 100).astype(np.int32)):
                #             time.sleep(0.04)
                #             print("setting target to", i/100.0)
                #             q_poseList[idx+13] = i/100.0
                #             h1arm.SetMotorPose(q_poseList, q_tau_ff)
                # break

                # print("q_poseList after setmoter: ", q_poseList)
                # if not is_first:
                #     for idx, val in enumerate(sol_q):
                #         if np.abs(val - last_sol_q[idx]) >= 1:
                #             sol_q[idx] = last_sol_q[idx]

                # q_poseList[13:27] = sol_q
                # q_poseList[13:20] = q_poseList[20:27]
                # q_poseList[13+4] = -q_poseList[13+4]
                # q_poseList[13+1] = -q_poseList[13+1]
                # q_poseList[13+2] = -q_poseList[13+2]

                # h1arm.SetMotorPose(q_poseList, q_tau_ff)

                # time.sleep(0.25)

                # if is_first:
                #     last_sol_q = sol_q
                #     is_first = False

                if flag:
                    q_poseList[13:27] = sol_q
                    q_tau_ff[13:27] = tau_ff
                else:
                    q_poseList[13:27] = armstate
                    q_tau_ff = np.zeros(35)

                # h1arm.SetMotorPose(q_poseList, q_tau_ff)

                # for i in range(100):
                #     time.sleep(0.1)
                #     q_poseList[18] += 0.005 # 18, 19, 26
                #     h1arm.SetMotorPose(q_poseList, q_tau_ff)
                # break

                if right_qpos is not None and left_qpos is not None:
                    # 4,5: index 6,7: middle, 0,1: pinky, 2,3: ring, 8,9: thumb
                    right_angles = [1.7 - right_qpos[i] for i in [4, 6, 2, 0]]
                    right_angles.append(1.2 - right_qpos[8])
                    right_angles.append(0.5 - right_qpos[9])

                    left_angles = [1.7 - left_qpos[i] for i in [4, 6, 2, 0]]
                    left_angles.append(1.2 - left_qpos[8])
                    left_angles.append(0.5 - left_qpos[9])
                    h1hand.crtl(right_angles, left_angles)

    except KeyboardInterrupt:
        print("Recording ended!")
    finally:
        # if video_writer is not None:
        #     video_writer.release()
        if rs_writer is not None:
            rs_writer.release()

        exit(0)
