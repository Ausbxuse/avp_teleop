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
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory

import cv2
import msgpack
import numpy as np
import zmq

from robot_control.robot_arm import H1ArmController
from robot_control.robot_arm_ik import Arm_IK
from robot_control.robot_hand import H1HandController
from utilities import VuerTeleop


def timed(func):
   def wrapper(*args, **kwargs):
       start = time.time()
       result = func(*args, **kwargs)
       elapsed = time.time() - start
       logger.debug(f"{func.__name__} took {elapsed*1000:.2f}ms")
       return result
   return wrapper

def run_with_retries(func, default_return=None, max_retries=3):
   """Execute a function with retry logic"""
   for attempt in range(max_retries):
       try:
           return func()
       except Exception as e:
           logger.warning(f"Error in {func.__name__}: {e} (attempt {attempt+1}/{max_retries})")
           time.sleep(0.1 * (2**attempt))  # Exponential backoff
   
   logger.error(f"Function {func.__name__} failed after {max_retries} attempts")
   return default_return

def monitor_resources():
   import psutil
   process = psutil.Process()
   memory = process.memory_info()
   cpu_percent = process.cpu_percent(interval=0.1)
   logger.debug(f"Memory: {memory.rss/1024/1024:.1f}MB, CPU: {cpu_percent}%")

# --------------------- Debug Logger Setup ---------------------
logger = logging.getLogger("robot_teleop")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# os.makedirs('logs', exist_ok=True)
# fh = logging.FileHandler(f"logs/robot_teleop_{time.strftime('%Y%m%d_%H%M%S')}.log")
# fh.setFormatter(formatter)
# logger.addHandler(fh)
# --------------------------------------------------------------

FREQ = 30
DELAY = 1 / FREQ
CHUNK_SIZE = 100

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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
    def __init__(self, dirname, buffer_size=100):
        self.buffer = []
        self.filepath = os.path.join(dirname, "ik_data.jsonl")
        self.async_writer = AsyncWriter(os.path.join(dirname, "ik_data.jsonl"))
        self.buffer_size = buffer_size  # Buffer size is no longer used here.

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
        try:
            if self.proc.poll() is None:  # if the process is still running
                logger.info("Sending SIGINT to the lidar process...")
                self.proc.send_signal(signal.SIGINT)
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()  # force kill after timeout
                    logger.info("Lidar process killed after timeout.")
        except Exception as e:
            logger.error(f"Error cleaning up lidar process: {e}")


class RobotDataWorker:
    def __init__(
        self, shared_data, kill_event, session_start_event, h1_shm_array, teleop_shm_array
    ):
        self.shared_data = shared_data
        self.kill_event = kill_event
        self.session_start_event = session_start_event
        self.h1_shm_array = h1_shm_array
        self.teleop_shm_array = teleop_shm_array
        self.h1_lock = Lock()
        self.teleoperator = VuerTeleop("inspire_hand.yml")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect("tcp://192.168.123.162:5556")
        self.socket.setsockopt(zmq.RCVTIMEO, 200)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.async_image_writer = AsyncImageWriter()

        # resetable vars
        self.frame_idx = 0
        self.last_robot_data = None
        self.robot_data_writer = None


    def dump_state(self, filename=None):
       """Dump current system state for debugging"""
       if filename is None:
           filename = f"debug_dump_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
       
       state = {
           "h1_data": self.h1_shm_array.copy(),
           "teleop_data": self.teleop_shm_array.copy(),
           "frame_idx": self.frame_idx if hasattr(self, 'frame_idx') else None,
           "timestamp": time.time()
       }
       
       with open(filename, 'wb') as f:
           pickle.dump(state, f)
       logger.info(f"State dumped to {filename}")

    def _sleep_until_mod33(self, time_curr):
        integer_part = int(time_curr)
        decimal_part = time_curr - integer_part
        ms_part = int(decimal_part * 1000) % 100

        next_ms_part = ((ms_part // 33) + 1) * 33 % 100
        hundred_ms_part = int(decimal_part * 10 % 10)
        if next_ms_part == 32:
            hundred_ms_part += 1

        next_capture_time = integer_part + next_ms_part / 1000 + hundred_ms_part / 10
        if (next_capture_time - time_curr) < 0:
            next_capture_time += 1
        time.sleep(next_capture_time - time_curr)

    def _recv_zmq_frame(self):
        compressed_data = b""
        while not self.kill_event.is_set():  # TODO: verify correctness
            logger.debug(f"start receving!")
            chunk = self.socket.recv()
            compressed_data += chunk
            if len(chunk) < 240000:  # Check for last chunk
                break

        try:
            data = zlib.decompress(compressed_data)
            frame_data = pickle.loads(data)
            # frame_data = msgpack.unpackb(data)
        except Exception as e:
            logger.error(f"Failed decompressing or unpickling frame data: {e}")
            return None, None

        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)  # np: (height, width)
        print(frame.shape)

        if frame is None:
            logger.error("Failed to decode frame!")
            return None, None
        
        width_each = frame.shape[1] // 4

        color_frame = frame[:, 0:width_each]
        depth_frame = frame[:, width_each:2*width_each]
        ir_left_frame = frame[:, 2*width_each:3*width_each]
        ir_right_frame = frame[:, 3*width_each:]
        return color_frame, depth_frame, ir_left_frame, ir_right_frame

    def teleop_update_thread(self):
        while not self.kill_event.is_set():
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                self.teleoperator.step()
            )
            # logger.debug(f"teleop step thread: {head_rmat}, {left_pose}, {right_pose}")
            self.teleop_shm_array[0:9] = head_rmat.flatten()
            self.teleop_shm_array[9:25] = left_pose.flatten()
            self.teleop_shm_array[25:41] = right_pose.flatten()
            self.teleop_shm_array[41:53] = np.array(left_qpos).flatten()
            self.teleop_shm_array[53:65] = np.array(right_qpos).flatten()
            time.sleep(1.0 / FREQ)

    def image_buffer_thread(self, image_queue):
        while not self.kill_event.is_set():
            try:
                frame = image_queue.get(timeout=0.1)
                np.copyto(self.teleoperator.img_array, frame)
                # logger.debug("image_buf_thread: copied frame")
            except queue.Empty:
                logger.debug("image_buf_thread: empty image")
                continue  
        logger.debug("Worker's image thread: recvd killevent")

    def get_robot_data(self, color_frame, depth_frame, time_curr):
        logger.debug(f"worker: starting to get robot data")
        color_filename = os.path.join(
            self.shared_data["dirname"], f"color/frame_{self.frame_idx:06d}.jpg"
        )
        depth_filename = os.path.join(
            self.shared_data["dirname"], f"depth/frame_{self.frame_idx:06d}.jpg"
        )
        if color_frame is not None and depth_frame is not None:
            self.async_image_writer.write_image(color_filename, color_frame)
            self.async_image_writer.write_image(depth_filename, depth_frame)
            logger.debug(
                f"Saved color frame to {color_filename} and depth frame to {depth_filename}"
            )
        else:
            logger.error(f"failed to save image {self.frame_idx}")

        with self.h1_lock:
            h1_data = self.h1_shm_array.copy()
        armstate = h1_data[0:14]
        legstate = h1_data[14:27]
        handstate = h1_data[27:39]
        imustate = h1_data[39:45]
        robot_data = {
            "time": time_curr,
            "arm_state": armstate.tolist(),
            "leg_state": legstate.tolist(),
            "hand_state": handstate.tolist(),
            "image": f"color/frame_{self.frame_idx:06d}.jpg",
            "depth": f"depth/frame_{self.frame_idx:06d}.jpg",
            "imu_omega": imustate[0:3].tolist(),
            "imu_rpy": imustate[3:6].tolist(),
            "ik_data": None,
            "lidar": None,
        }
        # logger.debug(f"worker: finish getting robot data")
        return robot_data

    def start(self):
        try:
            while True:
                logger.info("Worker: waiting for new session start (session_start_event).")
                self.session_start_event.wait()
                logger.info("Worker: starting new session.")
                self.run_session()
        finally:
            self.context.term()

    def _write_robot_data(self, color_frame, depth_frame, reuse=False):
        robot_data = self.get_robot_data(
            color_frame, depth_frame, time.time()
        )
        if reuse:
            self.last_robot_data["time"] = time.time()
            self.robot_data_writer.write(json.dumps(robot_data))
        else:
            if self.robot_data_writer is not None:
                self.robot_data_writer.write(json.dumps(robot_data))
        self.last_robot_data = robot_data
        self.frame_idx += 1

    def _send_image_to_teleoperator(self, color_frame):
        if color_frame is not None:
            resized_frame = cv2.resize(
                cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB), (1280, 720), interpolation=cv2.INTER_LINEAR
            )
            np.copyto(self.teleoperator.img_array, np.array(resized_frame))

    def _session_init(self):
        self.robot_data_writer = AsyncWriter(
            os.path.join(self.shared_data["dirname"], "robot_data.jsonl")
        )

        self.teleop_thread = threading.Thread(target=self.teleop_update_thread)
        self.teleop_thread.daemon = True
        self.teleop_thread.start()
        logger.info("RobotDataworker: teleop step started")

    def process_data(self):

        color_frame, depth_frame = self._recv_zmq_frame()
        # logger.debug("got frame")
        time_curr = time.time()
        self._send_image_to_teleoperator(color_frame)

        # logger.debug(f"Worker: got image")
        if self.is_first:
            self.is_first = False
            self._sleep_until_mod33(time.time())
            self.initial_capture_time = time.time()  # Store it as instance variable
            logger.debug(f"Worker: initial_capture_time is {self.initial_capture_time}")
            self._write_robot_data(color_frame, depth_frame)
            return

        next_capture_time = self.initial_capture_time + self.frame_idx * DELAY
        time_curr = time.time()
        logger.debug(
            f"[worker process] next_capture_time - time_curr: {next_capture_time - time_curr}"
        )

        if time_curr <= next_capture_time:
            time.sleep(next_capture_time - time_curr)
            self._write_robot_data(color_frame, depth_frame)
        else:
            logger.error(
                "worker process: runner did not finish within 33ms, reusing previous data"
            )
            if self.last_robot_data is not None:
                self._write_robot_data(color_frame, depth_frame, reuse=True)
            else:
                logger.error(
                    "worker process: no previous data available, generating null data"
                )
                self._write_robot_data(None, None, reuse=True)

    def run_session(self):
        self._session_init()
        self.is_first = True
        try:
            while not self.kill_event.is_set():
                logger.debug("Worker: entering main loop")
                self.process_data()

        except Exception as e:
            logger.error(f"robot_data_worker encountered an error: {e}")

        finally:
            logger.info("Worker process has exited.")
            # TODO: flush the buffer?
            self.teleop_thread.join(1)
            logger.info("Worker: teleop thread joined.")
            self.robot_data_writer.close()
            logger.info("Worker: writer closed.")
            self.reset()
            logger.info("Worker: closing async image writer.")
            if hasattr(self, 'async_image_writer'):
                self.async_image_writer.close()
            logger.info("Worker process has exited.")

    def reset(self):
        # TODO: finish rest
        self.frame_idx = 0
        self.initial_capture_time = None


# Teleop and datacollector
class RobotTaskmaster:
    def __init__(self, shared_data, h1_shm_array, teleop_shm_array, task_name, session_start_event, kill_event):
        self.task_name = task_name
        self.kill_event = kill_event
        self.session_start_event = session_start_event
        self.shared_data = shared_data
        self.h1_shm_array = h1_shm_array
        self.teleop_shm_array = teleop_shm_array

        self.teleop_lock = Lock()
        self.h1hand = H1HandController()
        self.h1arm = H1ArmController()
        self.arm_ik = Arm_IK()
        self.first = True
        self.lidar_proc = None
        self.ik_writer = None
        self.running = False
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
            + [np.pi ] * 2  # left wrists
            + [np.pi / 3] * 5
            + [np.pi ] * 2
        )
        if last_sol_q is not None and np.any(
            np.abs(last_sol_q - sol_q) > dynamic_thresholds
        ):
            logger.error("Master: ik movement too large!")
            return False
        if not ik_flag:
            q_poseList[13:27] = armstate
            q_tau_ff = np.zeros(35)
            logger.error("Master: ik flag false!")
            return False

        logger.debug("Master: preparing to set motor")
        self.h1arm.SetMotorPose(q_poseList, q_tau_ff)
        logger.debug("Master: motor set")

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
        try:
            while True:
                logger.info("Master: waiting to start")
                self.session_start_event.wait()
                logger.info("Master: start event recvd. clearing start event. starting session")
                self.run_session()
                logger.info("Master: merging data...")
                self.merge_data() # TODO: maybe a separate thread?
                logger.info("Master: merge finished. Preparing for a new run...")
                self.reset()
                logger.info("Master: reset finished")
        finally:
            logger.info("Master: finished")

    def get_h1_data(self):
        armstate, armv = self.h1arm.GetMotorState()
        legstate, _ = self.h1arm.GetLegState()
        handstate = self.h1hand.get_hand_state()
        imustate = self.h1arm.GetIMUState()

        # Also send data to worker through shared buffer
        with self.h1_lock:
            logger.debug("Master: h1 locking")
            self.h1_shm_array[0:14] = armstate
            self.h1_shm_array[14:27] = legstate
            self.h1_shm_array[27:39] = handstate
            self.h1_shm_array[39:42] = imustate.omega
            self.h1_shm_array[42:45] = imustate.rpy
        return armstate, armv

    def get_teleoperator_data(self):
        with self.teleop_lock:
            teleop_data = self.teleop_shm_array.copy()
        # logger.debug(f"Master: receving data : {teleop_data}")
        if np.all(teleop_data == 0):
            logger.debug(f"Master: not receving data yet: {teleop_data}")
            return False, None, None, None, None, None
        head_rmat = teleop_data[0:9].reshape(3, 3)
        left_pose = teleop_data[9:25].reshape(4, 4)
        right_pose = teleop_data[25:41].reshape(4, 4)
        left_qpos = teleop_data[41:53]
        right_qpos = teleop_data[53:65]
        return True, head_rmat, left_pose, right_pose, left_qpos, right_qpos

    
    def _session_init(self):
        self.lidar_proc = LidarProcess(self.shared_data["dirname"])
        self.lidar_proc.run()
        logger.debug("Master: lidar process started")
        self.running = True
        self.ik_writer = IKDataWriter(self.shared_data["dirname"])
        logger.debug("Master: getting teleop shm name")

    def run_session(self):
        self._session_init()
        last_sol_q = None
        logger.debug("Master: waiting for kill event")
        while not self.kill_event.is_set():
            logger.debug("Master: looping")
            # print("loop start",time.time())
            armstate, armv = self.get_h1_data()
            motor_time = time.time() # TODO: might be late here/ consider puting it before getmotorstate

            get_tv_success, head_rmat, left_pose, right_pose, left_qpos, right_qpos = self.get_teleoperator_data()

            if not get_tv_success:
                continue

            sol_q, tau_ff, ik_flag = self.arm_ik.ik_fun(left_pose, right_pose, armstate, armv)

            ik_time = time.time()
            # print("ik finish",time.time())

            logger.debug(f"Master: moving motor {sol_q}")
            if self.safelySetMotor(ik_flag, sol_q, last_sol_q, tau_ff, armstate, right_qpos, left_qpos):
                last_sol_q = sol_q
            else:
                continue

            logger.debug("Master: writing data")
            self.ik_writer.write_data(motor_time, ik_time, sol_q, tau_ff, head_rmat, left_pose, right_pose)

    def stop(self):
        self.running = False
        if self.lidar_proc is not None:
            self.lidar_proc.cleanup()
        logger.debug("Master: shutting down h1 contorllers...")
        self.h1arm.shutdown()
        self.h1hand.shutdown()
        logger.debug("Master: h1 controlleers shutdown")
        logger.info("Master: Stopping all threads ended!")

    def reset(self):
        logger.info("Master: Resetting RobotTaskmaster...")
        if self.running:
            self.stop()
        logger.info("Master: Clearing stop event...")
        # self.kill_event.clear()  # TODO: create a new one?

        self.h1hand.reset()
        self.h1arm.reset()
        self.first = True
        self.running = False

        self.h1_shm_array[:] = 0

        self.ik_writer = IKDataWriter(self.shared_data["dirname"])

        logger.info("RobotTaskmaster has been reset and is ready to start again.")

    def merge_data(self):
        if self.ik_writer is not None:
            self.ik_writer.close()
        merger = DataMerger(self.shared_data["dirname"])
        merger.merge_json()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Robot Teleoperation System")
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

    logger.info(f"#### (Task: {args.task_name}):")
    return args.task_name

def update_dir(shared_data, task_name):
    shared_data["dirname"] = time.strftime(f"demos/{task_name}/%Y%m%d_%H%M%S")
    os.makedirs(shared_data["dirname"], exist_ok=True)
    os.makedirs(os.path.join(shared_data["dirname"], "color"), exist_ok=True)
    os.makedirs(os.path.join(shared_data["dirname"], "depth"), exist_ok=True)

def setup_processes():
    task_name = parse_arguments()
    session_start_event = Event()
    kill_event = Event()
    manager = Manager()
    shared_data = manager.dict()

    h1_shm = shared_memory.SharedMemory(create=True, size=45 * np.dtype(np.float64).itemsize)
    h1_shm_array = np.ndarray((45,), dtype=np.float64, buffer=h1_shm.buf)

    teleop_shm = shared_memory.SharedMemory( create=True, size=65 * np.dtype(np.float64).itemsize)
    teleop_shm_array = np.ndarray((65,), dtype=np.float64, buffer=teleop_shm.buf)

    def run_taskmaster():
        taskmaster = RobotTaskmaster(shared_data, h1_shm_array, teleop_shm_array, task_name, session_start_event, kill_event)
        taskmaster.start()

    def run_dataworker():
        taskworker = RobotDataWorker(shared_data, kill_event, session_start_event, h1_shm_array, teleop_shm_array)
        taskworker.start()

    robot_data_proc = Process(target=run_dataworker)
    robot_data_proc.start()
    # TODO: fix inconsistent arm time (not strictly 33hz)
    taskmaster_proc = Process(target=run_taskmaster)
    taskmaster_proc.start()

    return task_name, kill_event, session_start_event,shared_data, h1_shm, teleop_shm, taskmaster_proc, robot_data_proc

def cleanup_processes(kill_event, taskmaster_proc, robot_data_proc):
    kill_event.set()
    logger.debug("Signaling processes to terminate...")
    
    logger.debug("Waiting for master process to terminate...")
    taskmaster_proc.join(timeout=3)
    
    logger.debug("Waiting for data process to terminate...")
    robot_data_proc.join(timeout=3)
    
    if taskmaster_proc.is_alive():
        logger.warning("Forcing termination of master process...")
        taskmaster_proc.terminate()
        taskmaster_proc.join(timeout=2)
    
    if robot_data_proc.is_alive():
        logger.warning("Forcing termination of data process...")
        robot_data_proc.terminate()
        robot_data_proc.join(timeout=2)

def main():
    # TODO: cleanup empty demo dirs
    task_name, kill_event, session_start_event, shared_data, h1_shm, teleop_shm, taskmaster_proc, robot_data_proc = setup_processes()
    logger.info("  Press 's' to start the taskmaster")
    logger.info("  Press 'q' to stop and merge data")
    try:
        while True:
            if sys.stdin.closed:  # TODO: why???
                logger.error("Standard input is closed. Continuing...")
                sys.stdin = open("/dev/tty")
                continue
            user_input = input("> ").lower()

            if user_input == "s":
                update_dir(shared_data, task_name)
                kill_event.clear()
                session_start_event.set()
                logger.info("Started taskmaster and dataworker")

            elif user_input == "q":
                logger.info("Clearing session start event and setting stop event")
                kill_event.set()
                session_start_event.clear() 
                logger.info("Ready to rerun!")

            elif user_input == "exit":
                logger.info("Exiting...")
                cleanup_processes(kill_event, taskmaster_proc, robot_data_proc)
                logger.debug("Data proc terminated")
                sys.exit(0)

            else:
                logger.info(
                    "Invalid. Use 's' to start, 'q' to stop/merge, 'exit' to quit."
                )

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Exiting...")
        cleanup_processes(kill_event, taskmaster_proc, robot_data_proc)
    finally:
        h1_shm.close()
        h1_shm.unlink()
        teleop_shm.close()
        teleop_shm.unlink()
        sys.exit(0)

def test_data_worker_main():
    session_start_event = Event()
    kill_event = Event()
    manager = Manager()
    shared_data = manager.dict()

    h1_shm = shared_memory.SharedMemory(create=True, size=45 * np.dtype(np.float64).itemsize)
    h1_shm_array = np.ndarray((45,), dtype=np.float64, buffer=h1_shm.buf)

    teleop_shm = shared_memory.SharedMemory( create=True, size=65 * np.dtype(np.float64).itemsize)
    teleop_shm_array = np.ndarray((65,), dtype=np.float64, buffer=teleop_shm.buf)

    def run_dataworker():
        taskworker = RobotDataWorker(shared_data, kill_event, session_start_event, h1_shm_array, teleop_shm_array)
        taskworker.start()

    robot_data_proc = Process(target=run_dataworker)
    kill_event.clear()
    session_start_event.set()

    robot_data_proc.start()


if __name__ == "__main__":
    main()
    # test_data_worker_main()
    # TODO: dirname use shared dic
