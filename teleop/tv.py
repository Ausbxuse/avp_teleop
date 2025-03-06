import argparse
import datetime
import json
import logging
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
import zlib
from multiprocessing import Event, Lock, Process, Queue, shared_memory
from queue import Empty, Full
from turtle import color

import cv2
import numpy as np
import zmq
from casadi import ne

from robot_control.robot_arm import H1ArmController
from robot_control.robot_arm_ik import Arm_IK
from robot_control.robot_hand import H1HandController
from utilities import VuerTeleop

# --------------------- Debug Logger Setup ---------------------
logger = logging.getLogger("robot_teleop")
logger.setLevel(logging.INFO)  # Default level; will be updated if --debug is passed.
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# --------------------------------------------------------------

FREQ = 30
DELAY = 1 / FREQ
CHUNK_SIZE = 100

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class IKDataWriter:
    def __init__(self, dirname, buffer_size=100):
        self.buffer = []
        self.filepath = os.path.join(dirname, "ik_data.jsonl")
        self.buffer_size = buffer_size
        self.lock = threading.Lock()

    def write_data(
        self,
        right_angles,
        left_angles,
        arm_time,
        ik_time,
        sol_q,
        tau_ff,
        head_rmat,
        left_pose,
        right_pose,
    ):
        entry = {
            "right_angles": right_angles,
            "left_angles": left_angles,
            "armtime": arm_time,
            "iktime": ik_time,
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
        logger.debug(f"Flushed IK data buffer to {self.filepath}")
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
    logger.debug(
        f"sleep_until_mod33: Sleeping until {next_capture_time} (current time: {time_curr})"
    )
    if (next_capture_time - time_curr) < 0:
        next_capture_time+=1
    time.sleep(next_capture_time - time_curr)


def recv_zmq_frame(socket, stop_event):
    compressed_data = b""
    while not stop_event.is_set():  # TODO: verify correctness
        chunk = socket.recv()
        compressed_data += chunk
        if len(chunk) < 120000:  # Check for last chunk
            break

    try:
        data = zlib.decompress(compressed_data)
        frame_data = pickle.loads(data)
    except Exception as e:
        logger.error(f"Error decompressing or unpickling frame data: {e}")
        return None, None

    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)  # np: (height, width)

    if frame is None:
        logger.error("[ERROR] Failed to decode frame!")
        return None, None

    color_frame = frame[:, : frame.shape[1] // 2]
    depth_frame = frame[:, frame.shape[1] // 2 :]
    return color_frame, depth_frame


def get_robot_data(
    dirname, frame_count, color_frame, depth_frame, h1_lock, h1_shm_array, time_curr
):
    color_filename = os.path.join(dirname, f"color/frame_{frame_count:06d}.jpg")
    depth_filename = os.path.join(dirname, f"depth/frame_{frame_count:06d}.jpg")
    if color_frame is not None and depth_frame is not None:
        cv2.imwrite(color_filename, color_frame)
        cv2.imwrite(depth_filename, depth_frame)
        logger.debug(
            f"Saved color frame to {color_filename} and depth frame to {depth_filename}"
        )

    with h1_lock:
        h1_data = h1_shm_array.copy()
    armstate = h1_data[0:14]
    legstate = h1_data[14:27]
    handstate = h1_data[27:39]
    imustate = h1_data[39:45]
    robot_data = {
        "time": time_curr,
        "arm_state": armstate.tolist(),
        "leg_state": legstate.tolist(),
        "hand_state": handstate.tolist(),
        "image": f"color/frame_{frame_count:06d}.jpg",
        "depth": f"depth/frame_{frame_count:06d}.jpg",
        "imu_omega": imustate[0:3].tolist(),
        "imu_rpy": imustate[3:6].tolist(),
        "ik_data": None,
        "lidar": None,
    }
    return robot_data


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

    logger.info("loading robot and IK data for merging.")
    robot_data_json_list = []
    with open(robot_data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                robot_data_json_list.append(json.loads(line))

    ik_data_list = []
    with open(ik_data_path, "r") as f:
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

    with open(output_path, "w") as f:
        json.dump(robot_data_json_list, f, indent=4)

    logger.info(f"Mergefile saved to {output_path}")


def profile(name, time_prev):
    time_curr = time.time()
    with open(f"profile.txt", "a") as file:
        file.write(f"{time_curr - time_prev}s:{name}\n")
    logger.debug(f"profile: {time_curr - time_prev}s for {name}")
    return time_curr


class LidarProcess:
    def __init__(self, dirname) -> None:
        self.program_cmd = [
            "./point_cloud_recorder",
            "./mid360_config.json",
            dirname + "/lidar",
        ]
        self.proc = None

    def run(self):
        self.proc = subprocess.Popen(
            self.program_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.info("LidarProcess started.")

    def cleanup(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:  # if the process is still running
            logger.info("Sending SIGINT to the lidar process...")
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()  # force kill after timeout
                logger.info("Lidar process killed after timeout.")


def teleop_update_thread(teleoperator, shm_name, stop_event):
    shm = shared_memory.SharedMemory(name=shm_name)
    teleop_array = np.ndarray((65,), dtype=np.float64, buffer=shm.buf)
    while not stop_event.is_set():
        head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
        teleop_array[0:9] = head_rmat.flatten()
        teleop_array[9:25] = left_pose.flatten()
        teleop_array[25:41] = right_pose.flatten()
        teleop_array[41:53] = np.array(left_qpos).flatten()
        teleop_array[53:65] = np.array(right_qpos).flatten()
        time.sleep(1.0 / FREQ)
    shm.close()

def image_buffer_thread(teleoperator, image_queue, stop_event):
    while not stop_event.is_set():
        try:
            frame = image_queue.get(timeout=0.1)
            np.copyto(teleoperator.img_array, frame)
        except Empty:
            continue  # No image available yet, loop again

def robot_data_worker(dirname, stop_event, start_time, h1_shm_array, teleop_shm_queue):
    local_teleoperator = VuerTeleop("inspire_hand.yml")
    teleop_shm = shared_memory.SharedMemory(create=True, size=65 * np.dtype(np.float64).itemsize)
    teleop_shm_queue.put(teleop_shm.name)
    teleop_thread = threading.Thread(
        target=teleop_update_thread, args=(local_teleoperator, teleop_shm.name, stop_event)
    )

    teleop_thread.daemon = True
    teleop_thread.start()

    image_queue = Queue(maxsize=10)
    image_thread = threading.Thread(
        target=image_buffer_thread, args=(local_teleoperator, image_queue, stop_event)
    )
    image_thread.daemon = True
    image_thread.start()

    robot_data_list = []
    log_filename = os.path.join(dirname, "robot_data.jsonl")

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.162:5556")

    frame_count = 0
    socket.setsockopt(zmq.RCVTIMEO, 200)
    socket.setsockopt(zmq.RCVHWM, 1)
    is_first = True
    h1_lock = Lock()

    try:
        while not stop_event.is_set():
            color_frame, depth_frame = recv_zmq_frame(socket, stop_event)
            time_curr = time.time()
            if color_frame is not None:
                resized_frame = cv2.resize(
                    color_frame, (1280, 720), interpolation=cv2.INTER_LINEAR
                )
                np.copyto(local_teleoperator.img_array, np.array(resized_frame))
            if is_first:
                is_first = False
                sleep_until_mod33(time_curr)
                initial_capture_time = time.time()
                logger.debug(f"initial_capture_time is {initial_capture_time}")
                robot_data = get_robot_data(
                    dirname,
                    frame_count,
                    color_frame,
                    depth_frame,
                    h1_lock,
                    h1_shm_array,
                    time.time(),
                )
                robot_data_list.append(robot_data)
                frame_count += 1
                continue


            next_capture_time = initial_capture_time + frame_count * DELAY
            time_curr = time.time()
            logger.debug(
                f"[worker process] next_capture_time - time_curr: {next_capture_time - time_curr}"
            )
            if time_curr <= next_capture_time:
                time.sleep(next_capture_time - time_curr)
                robot_data = get_robot_data(
                    dirname,
                    frame_count,
                    color_frame,
                    depth_frame,
                    h1_lock,
                    h1_shm_array,
                    time.time(),
                )
                robot_data_list.append(robot_data)
            else:
                logger.error(
                    "[ERROR] worker process: runner did not finish within 33ms, reusing previous data"
                )
                if "last_robot_data" in locals() and last_robot_data is not None:
                    last_robot_data["time"] = time.time()
                    robot_data_list.append(last_robot_data)
                else:
                    logger.error(
                        "[ERROR] worker process: no previous data available, generating null data"
                    )
                    robot_data = get_robot_data(
                        dirname,
                        frame_count,
                        None,
                        None,
                    h1_lock,
                    h1_shm_array,
                        time.time(),
                    )
                    robot_data_list.append(robot_data)
            last_robot_data = robot_data
            frame_count += 1

    except KeyboardInterrupt as e:
        logger.info(f"[INTR] keyboard interrupted: {e}")
    except Exception as e:
        logger.error(f"[ERROR] robot_data_worker encountered an error: {e}")

    finally:
        if robot_data_list:
            with open(log_filename, "a") as f:
                for data in robot_data_list:
                    json.dump(data, f)
                    f.write("\n")
            logger.debug(f"Flushed remaining robot data to {log_filename}")
        context.term()
        logger.info("robot_data_worker process has exited.")
        stop_event.set()
        teleop_thread.join()
        teleop_shm.close()
        teleop_shm.unlink()


# Teleop and datacollector
class RobotTaskmaster:
    def __init__(self, task_name):
        self.task_name = task_name
        self.stop_event = Event()
        self.teleop_lock = Lock()
        self.h1hand = H1HandController()
        self.h1arm = H1ArmController()
        self.arm_ik = Arm_IK()
        self.first = True
        self.robot_data_proc = None
        self.lidar_proc = None
        self.ik_writer = None
        self.running = False
        self.teleop_shm_queue = Queue()
        self.h1_shm_queue = Queue()
        self.dirname = time.strftime(f"demos/{self.task_name}/%Y%m%d_%H%M%S")
        self.h1_shm = shared_memory.SharedMemory(create=True, size=45 * np.dtype(np.float64).itemsize)
        self.h1_shm_array = np.ndarray((45,), dtype=np.float64, buffer=self.h1_shm.buf)
        self.h1_lock = Lock()


    def safelySetMotor(
        self, ik_flag, sol_q, last_sol_q, tau_ff, armstate, right_qpos, left_qpos
    ):
        q_poseList = np.zeros(35)
        q_tau_ff = np.zeros(35)
        q_poseList[13:27] = sol_q
        q_tau_ff[13:27] = tau_ff  # WARN: untested!
        dynamic_thresholds = np.array(
            [np.pi / 3] * 5  # left shoulder and elbow
            + [np.pi / 1.5] * 2  # left wrists
            + [np.pi / 3] * 5
            + [np.pi / 1.5] * 2
        )
        if last_sol_q is not None and np.any(
            np.abs(last_sol_q - sol_q) > dynamic_thresholds
        ):
            logger.error("[ERROR] ik movement too large!")
            return False
        if not ik_flag:
            q_poseList[13:27] = armstate
            q_tau_ff = np.zeros(35)
            logger.error("[ERROR] ik flag false!")
            return False

        if np.any(np.abs(armstate - sol_q) > dynamic_thresholds) and self.first:
            self.first = False
            intermedia_sol_q = np.array(armstate)

            logger.error("[ERROR] slowing for large movement!")
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
            # self.h1hand.ctrl(right_hand_angles, left_hand_angles)
        return True

    def start(self):
        self.running = True
        os.makedirs(self.dirname)
        os.makedirs(os.path.join(self.dirname, "color"))
        os.makedirs(os.path.join(self.dirname, "depth"))
        self.lidar_proc = LidarProcess(self.dirname)
        self.ik_writer = IKDataWriter(self.dirname)
        self.lidar_proc.run()

        teleop_shm_name = self.teleop_shm_queue.get()
        self.teleop_shm = shared_memory.SharedMemory(name=teleop_shm_name)
        self.teleop_shm_array = np.ndarray((65,), dtype=np.float64, buffer=self.teleop_shm.buf)

        right_hand_angles = None
        left_hand_angles = None
        last_sol_q = None
        first = True
        while not self.stop_event.is_set():
            if first:
                first = False
                time.sleep(1)
            # print("loop start",time.time())
            armstate, armv = self.h1arm.GetMotorState()
            legstate, _ = self.h1arm.GetLegState()
            handstate = self.h1hand.get_hand_state()
            imustate = self.h1arm.GetIMUState()
            with self.h1_lock:
                self.h1_shm_array[0:14] = armstate
                self.h1_shm_array[14:27] = legstate
                self.h1_shm_array[27:39] = handstate
                self.h1_shm_array[39:42] = imustate.omega
                self.h1_shm_array[42:45] = imustate.rpy

            motor_time = time.time()
            with self.teleop_lock:
                teleop_data = self.teleop_shm_array.copy()
            # print("tv finish",time.time())
            head_rmat = teleop_data[0:9].reshape(3, 3)
            left_pose = teleop_data[9:25].reshape(4, 4)
            right_pose = teleop_data[25:41].reshape(4, 4)
            left_qpos = teleop_data[41:53]
            right_qpos = teleop_data[53:65]
            sol_q, tau_ff, ik_flag = self.arm_ik.ik_fun(
                left_pose, right_pose, armstate, armv
            )
            ik_time = time.time()
            # print("ik finish",time.time())
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

            self.ik_writer.write_data(
                right_hand_angles,
                left_hand_angles,
                motor_time,
                ik_time,
                sol_q,
                tau_ff,
                head_rmat,
                left_pose,
                right_pose,
            )

    def stop(self):
        self.running = False
        self.stop_event.set()
        if self.robot_data_proc is not None and self.robot_data_proc.is_alive():
            self.robot_data_proc.join(timeout=30)  # FIXME: times a very long time to write
        if self.lidar_proc is not None:
            self.lidar_proc.cleanup()
        if self.ik_writer is not None:
            self.ik_writer.close()
        self.h1arm.shutdown()
        self.h1hand.shutdown()
        logger.info("Stopping all threads ended!")

    def reset(self):
        logger.info("Resetting robotmaster")
        if self.running:
            self.stop()
        self.stop_event.clear()
        self.h1hand.reset()
        self.h1arm.reset()
        self.first = True
        self.running = False
        logger.info("RobotTaskmaster has been reset and is ready to start again.")

    def merge_data(self):
        ik_filepath = os.path.join(self.dirname, "ik_data.jsonl")
        motor_filepath = os.path.join(self.dirname, "robot_data.jsonl")
        output_filepath = os.path.join(self.dirname, "merged_data.jsonl")
        lidar_filepath = os.path.join(self.dirname, "lidar")
        merge_data_to_pkl(motor_filepath, ik_filepath, lidar_filepath, output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Teleoperation System")
    # TODO: cleanup empty demo dirs
    parser.add_argument(
        "--task_name",
        type=str,
        default="default_task",
        help="Name of the task for data collection (default: default_task)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging output"
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")

    print(f"Robo master control (Task: {args.task_name}):")
    print("  Press 's' to start the taskmaster")
    print("  Press 'q' to stop and merge data")

    # Although a teleoperator is passed here for API compatibility, it is now unused in the main process.
    task_thread = None
    taskmaster = RobotTaskmaster(args.task_name)

    def run_taskmaster():
        taskmaster.start()

    robot_data_proc = Process(
        target=robot_data_worker,
        args=(
            taskmaster.dirname,
            taskmaster.stop_event,
            time.time(),
            taskmaster.h1_shm_array,
            taskmaster.teleop_shm_queue,
        ),
    )
    robot_data_proc.start()

    try:
        while True:
            if sys.stdin.closed:  # TODO: why???
                logger.error("Standard input is closed. Continuing...")
                sys.stdin = open("/dev/tty")
                continue
            user_input = input("> ").lower()

            if user_input == "s" and not taskmaster.running:
                logger.info("Starting taskmaster...")
                task_thread = threading.Thread(target=run_taskmaster)
                task_thread.daemon = True
                task_thread.start()

            elif user_input == "q" and taskmaster.running:
                if task_thread is not None:
                    task_thread.join(timeout=1)
                logger.info("Stopping taskmaster")
                taskmaster.stop()
                logger.info("merging data")
                taskmaster.merge_data()
                print("Done! Press 's' to start again or type 'exit' to quit.")
                logger.info("Done! Press 's' to start again or type 'exit' to quit.")
                taskmaster = RobotTaskmaster(args.task_name)

            elif user_input == "exit":
                print("Exiting...")
                logger.info("Exiting...")
                if taskmaster.running and task_thread is not None:
                    taskmaster.stop()
                    task_thread.join(timeout=1)
                sys.exit(0)

            else:
                print(
                    "Invalid command. Use 's' to start, 'q' to stop/merge, or 'exit' to quit."
                )
                logger.info(
                    "Invalid command. Use 's' to start, 'q' to stop/merge, or 'exit' to quit."
                )

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting...")
        logger.info("KeyboardInterrupt detected. Exiting...")
        if taskmaster.running and task_thread is not None:
            taskmaster.stop()
            taskmaster.merge_data()
            task_thread.join(timeout=1)
        sys.exit(0)
