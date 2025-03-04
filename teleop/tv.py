import argparse
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
from multiprocessing import Event, Lock, Process

import cv2
import numpy as np
import zmq

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pickle

from robot_control.robot_arm import H1ArmController
from robot_control.robot_arm_ik import Arm_IK
from robot_control.robot_hand import H1HandController
from utilities import VuerTeleop

FREQ = 30
DELAY = 1 / FREQ
CHUNK_SIZE = 100


import json
import os
import threading


class IKDataWriter:
    def __init__(self, dirname, buffer_size=100):
        self.buffer = []
        self.filepath = os.path.join(dirname, "ik_data.json")
        self.buffer_size = buffer_size
        self.lock = threading.Lock()

    def write_data(
        self,
        right_angles,
        left_angles,
        armtime,
        iktime,
        sol_q,
        tau_ff,
        head_rmat,
        left_pose,
        right_pose,
    ):
        entry = {
            "right_angles": right_angles,
            "left_angles": left_angles,
            "armtime": armtime,
            "iktime": iktime,
            "sol_q": sol_q.tolist(),
            "tau_ff": tau_ff.tolist(),
            "head_rmat": head_rmat.tolist(),
            "left_pose": left_pose.tolist(),
            "right_pose": right_pose.tolist(),
        }
        self.buffer.append(entry)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        with open(self.filepath, "a") as f:
            for entry in self.buffer:
                f.write(json.dumps(entry) + "\n")
        self.buffer.clear()

    def close(self):
        """Flush any remaining entries."""
        with self.lock:
            if self.buffer:
                self.flush()


def sleep_until_mod33(time_curr):
    integer_part = int(time_curr)
    decimal_part = time_curr - integer_part
    ms_part = int(decimal_part * 1000) % 1000

    next_ms_part = ((ms_part // 33) + 1) * 33 % 1000

    next_capture_time = integer_part + next_ms_part / 1000
    time.sleep(next_capture_time - time_curr)


def recv_zmq_frame(socket, stop_event):
    compressed_data = b""
    while not stop_event.is_set():  # TODO: verify correctness
        chunk = socket.recv()
        compressed_data += chunk
        if len(chunk) < 120000:  # Check for last chunk
            break

    data = zlib.decompress(compressed_data)
    frame_data = pickle.loads(data)
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)  # np: (height, width)

    if frame is None:
        print("[ERROR] Failed to decode frame!")
        return None, None

    color_frame = frame[:, : frame.shape[1] // 2]
    depth_frame = frame[:, frame.shape[1] // 2 :]
    return color_frame, depth_frame


def get_robot_data(dirname, frame_count, color_frame, depth_frame, h1arm, h1hand):
    color_filename = os.path.join(dirname, f"color/frame_{frame_count:06d}.jpg")
    depth_filename = os.path.join(dirname, f"depth/frame_{frame_count:06d}.jpg")
    if color_frame is not None and depth_frame is not None:
        cv2.imwrite(color_filename, color_frame)
        cv2.imwrite(depth_filename, depth_frame)

    armstate, _ = h1arm.GetMotorState()
    legstate, _ = h1arm.GetLegState()
    handstate = h1hand.get_hand_state()
    imustate = h1arm.GetIMUState()
    robot_data = {
        "time": time.time(),
        "arm_state": armstate.tolist(),
        "leg_state": legstate.tolist(),
        "hand_state": handstate.tolist(),
        "image": f"color/frame_{frame_count:06d}.jpg",
        "depth": f"depth/frame_{frame_count:06d}.jpg",
        "imu_omega": imustate.omega,
        "imu_rpy": imustate.rpy,
        "ik_data": None,
        "lidar": None,
    }
    return robot_data


def robot_data_worker(
    dirname, stop_event, start_time, h1arm, h1hand, teleop_lock, teleoperator
):
    robot_data_list = []
    log_filename = os.path.join(dirname, "robot_data.txt")

    is_first = True

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.162:5556")

    frame_count = 0
    next_capture_time = start_time
    socket.setsockopt(zmq.RCVTIMEO, 200)
    try:
        while not stop_event.is_set():
            color_frame, depth_frame = recv_zmq_frame(socket, stop_event)
            if color_frame is not None:
                resized_frame = cv2.resize(
                    color_frame, (1280, 720), interpolation=cv2.INTER_LINEAR
                )
                with teleop_lock:
                    np.copyto(teleoperator.img_array, np.array(resized_frame))

            time_curr = time.time()
            if is_first is True:
                sleep_until_mod33(time_curr)
                is_first = False

            time_curr = time.time()
            if time_curr < next_capture_time:
                time.sleep(next_capture_time - time_curr)
            else:
                robot_data = get_robot_data(
                    dirname, frame_count, color_frame, depth_frame, h1arm, h1hand
                )
                robot_data_list.append(robot_data)
                frame_count += 1
                next_capture_time += DELAY
                if len(robot_data_list) >= CHUNK_SIZE:
                    with open(log_filename, "a") as f:
                        for data in robot_data_list:
                            json.dump(data, f)
                            f.write("\n")
                    robot_data_list = []

    except KeyboardInterrupt as e:
        print(f"[INTR] keyboard interrupted: {e}")
    except Exception as e:
        print(f"[ERROR] robot_data_worker encountered an error: {e}")

    finally:
        if robot_data_list:
            with open(log_filename, "a") as f:
                for data in robot_data_list:
                    json.dump(data, f)
                    f.write("\n")
        context.term()
        print("[INFO] robot_data_worker process has exited.")


def ik_is_ready(ik_data_list, time_key):
    closest_ik_entry = min(ik_data_list, key=lambda x: abs(x["armtime"] - time_key))
    if abs(closest_ik_entry["armtime"] - time_key) > DELAY / 2:
        return False, None
    return True, closest_ik_entry


def lidar_is_ready(lidar_time_list, time_key):
    closest_lidar_entry = min(lidar_time_list, key=lambda x: abs(x - time_key))
    if abs(closest_lidar_entry - time_key) > DELAY / 2:
        return False, None
    return True, closest_lidar_entry


def merge_data_to_pkl(robot_data_path, ik_data_path, lidar_data_path, output_path):
    lidar_time_list = []

    lidar_files = [
        f
        for f in os.listdir(lidar_data_path)
        if os.path.isfile(os.path.join(lidar_data_path, f))
    ]

    for lidar_file_name in lidar_files:
        time_parts = lidar_file_name.split(".")[0:2]
        lidar_time_list.append(float(time_parts[0] + "." + time_parts[1]))

    print("loading")
    with open(robot_data_path, "r") as f:
        robot_data_json_list = json.load(f)

    with open(ik_data_path, "r") as f:
        ik_data_list = json.load(f)

    ik_data_dict = {entry["armtime"]: entry for entry in ik_data_list}
    robot_data_dict = {entry["time"]: entry for entry in robot_data_json_list}

    if ik_data_list[0]["armtime"] > robot_data_json_list[0]["time"]:
        last_robot_data = None
    else:
        last_robot_data = ik_data_list[0]

    for motor_entry in robot_data_json_list:
        time_key = motor_entry["time"]
        ik_ready_flag, closest_ik_entry = ik_is_ready(ik_data_list, time_key)
        if ik_ready_flag and closest_ik_entry is not None:
            robot_data_dict[time_key]["ik_data"] = ik_data_dict[
                closest_ik_entry["armtime"]
            ]
            last_robot_data = robot_data_dict[time_key]["ik_data"]
        else:
            robot_data_dict[time_key]["ik_data"] = last_robot_data

        # merge lidar path
        lidar_ready_flag, closest_lidar_time = lidar_is_ready(lidar_time_list, time_key)
        if lidar_ready_flag:
            robot_data_dict[time_key]["lidar"] = os.path.join(
                "lidar", f"{closest_lidar_time}.pcd"
            )

    # with open(output_path, "wb") as f:
    #     pickle.dump(robot_data_list, f)
    with open(output_path, "w") as f:
        json.dump(robot_data_json_list, f, indent=4)

    print(f"Mergefile saved to {output_path}")


def profile(name, time_prev):
    time = datetime.datetime.now()
    with open(f"profile.txt", "a") as file:
        file.write(f"{time - time_prev}s:{name}\n")
    return time


class LidarProcess:
    def __init__(self, dirname) -> None:
        self.program_cmd = [
            "./point_cloud_recorder",
            "./mid360_config.json",
            dirname + "/lidar",
        ]

    def run(self):
        self.proc = subprocess.Popen(
            self.program_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def cleanup(self):
        if self.proc.poll() is None:  # if the process is still running
            print("Sending SIGINT to the lidar process...")
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()  # force kill if needed


# Teleop and datacollector
class RobotTaskmaster:
    def __init__(self, task_name, stop_event) -> None:
        self.dirname = time.strftime(f"demos/{task_name}/%Y%m%d_%H%M%S")
        os.makedirs(self.dirname)
        os.makedirs(os.path.join(self.dirname, "color"))
        os.makedirs(os.path.join(self.dirname, "depth"))
        self.lidar_proc = LidarProcess(self.dirname)
        self.stop_event = stop_event
        self.teleop_lock = Lock()
        self.h1hand = H1HandController()
        self.h1arm = H1ArmController()
        self.teleoperator = VuerTeleop("inspire_hand.yml")
        start_time = time.time()
        self.robot_data_proc = Process(
            target=robot_data_worker,
            args=(
                self.dirname,
                self.stop_event,
                start_time,
                self.h1arm,
                self.h1hand,
                self.teleop_lock,
                self.teleoperator,
            ),
        )
        self.ik_writer = IKDataWriter(self.dirname)
        self.arm_ik = Arm_IK()
        self.first = True

    def safelySetMotor(
        self, ik_flag, sol_q, last_sol_q, tau_ff, armstate, right_qpos, left_qpos
    ):
        q_poseList = np.zeros(35)
        q_tau_ff = np.zeros(35)
        q_poseList[13:27] = sol_q
        q_tau_ff[13:27] = tau_ff  # WARN: untested!
        dynamic_thresholds = np.array(
            [np.pi / 3] * 5  # left shoulder and elbo
            + [np.pi / 1.5] * 2  # left writs
            + [np.pi / 3] * 5
            + [np.pi / 1.5] * 2
        )
        if last_sol_q is not None and np.any(
            np.abs(last_sol_q - sol_q) > dynamic_thresholds
        ):
            print("[ERROR] ik movement too large!")
            return False
        if not ik_flag:
            q_poseList[13:27] = armstate
            q_tau_ff = np.zeros(35)
            print("[ERROR] ik flag false!")
            return False

        if np.any(np.abs(armstate - sol_q) > dynamic_thresholds) and self.first:
            self.first = False
            intermedia_sol_q = np.array(armstate)

            print("[ERROR] slowing for large movement!")
            while np.any(np.abs(sol_q - intermedia_sol_q) > np.pi / 90):
                step_sizes = (sol_q - intermedia_sol_q) / 50

                intermedia_sol_q += step_sizes
                q_poseList[13:27] = intermedia_sol_q
                self.h1arm.SetMotorPose(q_poseList, q_tau_ff)
                time.sleep(0.01)  # Small delay for smooth motion
            q_poseList[13:27] = sol_q
            self.h1arm.SetMotorPose(q_poseList, q_tau_ff)
        else:
            self.h1arm.SetMotorPose(q_poseList, q_tau_ff)

        if right_qpos is not None and left_qpos is not None:
            right_hand_angles = [1.7 - right_qpos[i] for i in [4, 6, 2, 0]]
            right_hand_angles.append(1.2 - right_qpos[8])
            right_hand_angles.append(0.5 - right_qpos[9])

            left_hand_angles = [1.7 - left_qpos[i] for i in [4, 6, 2, 0]]
            left_hand_angles.append(1.2 - left_qpos[8])
            left_hand_angles.append(0.5 - left_qpos[9])
            self.h1hand.ctrl(right_hand_angles, left_hand_angles)
        return True

    def start(self):
        self.lidar_proc.run()
        self.robot_data_proc.start()

        right_hand_angles = None
        left_hand_angles = None
        last_sol_q = None
        while not self.stop_event.is_set():
            # profile("Main loop started")
            armstate, armv = self.h1arm.GetMotorState()
            # profile("get arm finished")
            motor_time = time.time()
            # profile("before teleop step")
            # TODO: maybe thread might be faster
            with self.teleop_lock:
                head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                    self.teleoperator.step()
                )
            # profile("teleop finished")
            sol_q, tau_ff, ik_flag = self.arm_ik.ik_fun(
                left_pose, right_pose, armstate, armv
            )
            if self.safelySetMotor(
                ik_flag,
                sol_q,
                last_sol_q,
                tau_ff,
                armstate,
                right_qpos,
                left_qpos,
            ):
                last_sol_q = sol_q
            else:
                continue
            # profile("ik finished")

            self.ik_writer.write_data(  # TODO:  inefficient!
                right_hand_angles,
                left_hand_angles,
                motor_time,
                time.time(),
                sol_q,
                tau_ff,
                head_rmat,
                left_pose,
                right_pose,
            )

    def stop(self):
        self.stop_event.set()
        if self.robot_data_proc is not None and self.robot_data_proc.is_alive():
            self.robot_data_proc.join(2)
            self.robot_data_proc.terminate()
        self.lidar_proc.cleanup()
        self.ik_writer.close()
        print("Recording ended!")

    def merge_data(self):
        ik_filepath = os.path.join(self.dirname, "ik_data.json")
        motor_filepath = os.path.join(self.dirname, "robot_data.txt")
        output_filepath = os.path.join(self.dirname, "merged_data.json")
        lidar_filepath = os.path.join(self.dirname, "lidar")
        merge_data_to_pkl(motor_filepath, ik_filepath, lidar_filepath, output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot Teleoperation System')
    # TODO: cleanup empty demo dirs
    parser.add_argument('--task_name', type=str, default="default_task", 
                        help='Name of the task for data collection (default: default_task)')
    args = parser.parse_args()
    
    stop_event = Event()
    taskmaster = RobotTaskmaster(args.task_name, stop_event)
    print(f"Robo master control (Task: {args.task_name}):")
    print("  Press 's' to start the taskmaster")
    print("  Press 'q' to stop and merge data")

    def run_taskmaster():
        global running
        running = True
        taskmaster.start()

    try:
        while True:
            user_input = input("> ")
            if user_input.lower() == "s" and not running:
                print("Starting taskmaster...")
                task_thread = threading.Thread(target=run_taskmaster)
                task_thread.daemon = True
                task_thread.start()
            elif user_input.lower() == "q" and running:
                print("Stopping taskmaster and merging data...")
                running = False
                taskmaster.stop()
                taskmaster.merge_data()
                print("Done! Press 's' to start again or Ctrl+C to exit")
            elif user_input.lower() == "q" and not running:
                print("Exiting...")
                break
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if running:
            taskmaster.stop()
            taskmaster.merge_data()
