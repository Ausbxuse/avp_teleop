import argparse
import json
import logging
import os
import pickle
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import zlib
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory
from typing import Any, Dict, Optional, Tuple

import cv2
import msgpack
import numpy as np
import psutil
import zmq
from robot_control.robot_arm import H1ArmController
from robot_control.robot_arm_ik import Arm_IK
from robot_control.robot_hand import H1HandController
from utilities import (AsyncImageWriter, AsyncWriter, DataMerger, IKDataWriter,
                       VuerTeleop)


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
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': "\033[37m",    # white
        'INFO': "\033[32m",     # green
        'WARNING': "\033[33m",  # yellow
        'ERROR': "\033[31m",    # red
        'CRITICAL': "\033[41m", # red background
    }
    RESET = "\033[0m"

    def format(self, record):
        original_levelname = record.levelname
        if original_levelname in self.COLORS:
            record.levelname = f"{self.COLORS[original_levelname]}{original_levelname}{self.RESET}"
        formatted = super().format(record)
        record.levelname = original_levelname
        return formatted

logger = logging.getLogger("robot_teleop")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s")
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

# create a tv.step() thread and request image 
class RobotDataWorker:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        self.kill_event = shared_data["kill_event"]
        self.session_start_event = shared_data["session_start_event"]
        self.h1_shm_array = shared_data["h1_shm_array"]
        self.teleop_shm_array = shared_data["teleop_shm_array"]
        self.end_event = shared_data["end_event"] # TODO: redundent
        self.h1_lock = Lock()
        self.teleoperator = VuerTeleop("inspire_hand.yml")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://192.168.123.162:5556")
        # self.socket.setsockopt(zmq.RCVTIMEO, 200)
        # self.socket.setsockopt(zmq.RCVHWM, 1)
        # self.socket.setsockopt(zmq.CONFLATE, 1)
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

    def _recv_zmq_frame(self) -> Tuple[Any, Any, Any, Any]:
        color_size = 480 * 640 * 3
        depth_size = 480 * 640
        ir_size = 480 * 640

        self.socket.send(b"get_frame")
        # frame_bytes = self.socket.recv()
        parts = self.socket.recv_multipart()
        if len(parts) < 2:
            logger.error("Incomplete frame data received.")
            return None, None, None, None

        jpg_bytes = parts[0]
        depth_bytes = parts[1]

        jpg_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
        combined_img = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
        if combined_img is not None:
            cv2.imshow("Combined Color and IR", combined_img)
        else:
            logger.error("Failed to decode JPEG frame.")

        try:
            depth_image = pickle.loads(zlib.decompress(depth_bytes))
        except Exception:
            logger.error("Worker: zmq depth pickled fail")

        height, width, channels = combined_img.shape

        if width % 3 != 0:
            logger.error("Unexpected combined image width:", width)
            return None, None, None, None

        single_width = width // 3

        color_image = combined_img[:, :single_width, :]
        ir_left_image  = combined_img[:, single_width:2*single_width, :]
        ir_right_image = combined_img[:, 2*single_width:, :]
        
        return color_image, depth_image, ir_left_image, ir_right_image

    def teleop_update_thread(self):
        while not self.kill_event.is_set():
            # logger.info("Worker: tp_thread: stepping!")
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                self.teleoperator.step()
            )
            # logger.debug(f"teleop step thread: {head_rmat}, {left_pose}, {right_pose}")
            self.teleop_shm_array[0:9] = head_rmat.flatten()
            self.teleop_shm_array[9:25] = left_pose.flatten()
            self.teleop_shm_array[25:41] = right_pose.flatten()
            self.teleop_shm_array[41:53] = np.array(left_qpos).flatten()
            self.teleop_shm_array[53:65] = np.array(right_qpos).flatten()
            time.sleep(DELAY)

    def get_robot_data(self, time_curr):
        logger.debug(f"worker: starting to get robot data")
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
            "image": f"color/frame_{self.frame_idx:06d}.npy",
            "depth": f"depth/frame_{self.frame_idx:06d}.npy",
            "imu_omega": imustate[0:3].tolist(),
            "imu_rpy": imustate[3:6].tolist(),
            "ik_data": None,
            "lidar": None,
        }
        # logger.debug(f"worker: finish getting robot data")
        return robot_data

    def start(self):
        # logger.debug(f"Worker: Process ID (PID) {os.getpid()}")
        try:
            while True:
                logger.info("Worker: waiting for new session start (session_start_event).")
                self.session_start_event.wait()
                logger.info("Worker: starting new session.")
                self.run_session()
                self.async_image_writer.close()
                self.async_image_writer = AsyncImageWriter()
        finally:
            logger.info("worker: ending")
            self.socket.close()
            self.context.term()
            self.teleoperator.shutdown()

    # def _write_image_data(self, color_frame, depth_frame):
    #     logger.debug("Worker: writing robot data")
    #
    #     color_filename = os.path.join(
    #         self.shared_data["dirname"], f"color/frame_{self.frame_idx:06d}.jpg"
    #     )
    #     depth_filename = os.path.join(
    #         self.shared_data["dirname"], f"depth/frame_{self.frame_idx:06d}.jpg"
    #     )
    #
    #     if color_frame is not None and depth_frame is not None:
    #         self.async_image_writer.write_image(color_filename, color_frame)
    #         self.async_image_writer.write_image(depth_filename, depth_frame)
    #         logger.debug(
    #             f"Saved color frame to {color_filename} and depth frame to {depth_filename}"
    #         )
    #     else:
    #         logger.error(f"failed to save image {self.frame_idx}")
    #

    def _write_image_data(self, color_frame, depth_frame):
        logger.debug("Worker: writing robot data")

        color_filename = os.path.join(
            self.shared_data["dirname"], f"color/frame_{self.frame_idx:06d}.npy"
        )
        depth_filename = os.path.join(
            self.shared_data["dirname"], f"depth/frame_{self.frame_idx:06d}.npy"
        )

        if color_frame is not None and depth_frame is not None:
            # Pickle and compress each frame.
            compressed_color = zlib.compress(pickle.dumps(color_frame))
            compressed_depth = zlib.compress(pickle.dumps(depth_frame))

            # Save the compressed data to file.
            self.async_image_writer.write_image(color_filename, compressed_color)
            self.async_image_writer.write_image(depth_filename, compressed_depth)
            logger.debug(
                f"Saved color frame to {color_filename} and depth frame to {depth_filename}"
            )
        else:
            logger.error(f"failed to save image {self.frame_idx}")

    def _write_robot_data(self, color_frame, depth_frame, reuse=False):
        self._write_image_data(color_frame,depth_frame)

        robot_data = self.get_robot_data(time.time())

        if reuse:
            self.last_robot_data["time"] = time.time()
            self.robot_data_writer.write(json.dumps(robot_data))
        else:
            if self.robot_data_writer is not None:
                self.robot_data_writer.write(json.dumps(robot_data))
        self.last_robot_data = robot_data
        self.frame_idx += 1

    def _send_image_to_teleoperator(self, ir_left_frame, ir_right_frame):
        ir_left_frame = cv2.cvtColor(ir_left_frame, cv2.COLOR_GRAY2BGR)
        ir_right_frame = cv2.cvtColor(ir_right_frame, cv2.COLOR_GRAY2BGR)
        if ir_left_frame is not None and ir_right_frame is not None:
            combined_ir_frame = np.hstack((ir_left_frame, ir_right_frame))
            # print(combined_ir_frame.shape)
            resized_frame = cv2.resize(
                combined_ir_frame, (1280, 720), interpolation=cv2.INTER_LINEAR
            )
            np.copyto(self.teleoperator.img_array, np.array(resized_frame))

    def _session_init(self):
        if "dirname" not in self.shared_data:
            logger.error("Worker: failed to get dirname")
            exit(-1)
        self.robot_data_writer = AsyncWriter(
            os.path.join(self.shared_data["dirname"], "robot_data.jsonl")
        )

        self.teleop_thread = threading.Thread(target=self.teleop_update_thread, daemon=True)
        self.teleop_thread.start()
        logger.info("RobotDataworker: teleop step started")

    def process_data(self):

        logger.debug("request frame")
        color_frame, depth_frame, ir_left_frame, ir_right_frame = self._recv_zmq_frame()
        logger.debug("got frame")
        self._send_image_to_teleoperator(ir_left_frame, ir_right_frame)
        time_curr = time.time()

        # logger.debug(f"Worker: got image")
        if self.is_first:
            self.is_first = False
            self._sleep_until_mod33(time.time())
            self.initial_capture_time = time.time()
            self._write_robot_data(color_frame, depth_frame)
            logger.debug(f"Worker: initial_capture_time is {self.initial_capture_time}")
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
                self.robot_data_writer.close()
                self.robot_data_writer = AsyncWriter(
                    os.path.join(self.shared_data["dirname"], "robot_data.jsonl")
                )


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
                logger.info("Worker: async image writer closed.")
            logger.info("Worker process has exited.")

    def reset(self):
        self.frame_idx = 0
        self.initial_capture_time = None


# Teleop and datacollector
# Starts lidar process and robot arm/hand controllers
class RobotTaskmaster:
    def __init__(self, task_name, shared_data):
        self.task_name = task_name

        self.shared_data = shared_data
        self.kill_event = shared_data["kill_event"]
        self.session_start_event = shared_data["session_start_event"]
        self.h1_shm_array = shared_data["h1_shm_array"]
        self.teleop_shm_array = shared_data["teleop_shm_array"]
        self.failure_event = shared_data["failure_event"] # TODO: redundent
        self.end_event = shared_data["end_event"] # TODO: redundent

        self.teleop_lock = Lock()
        try:
            self.h1hand = H1HandController()
            self.h1arm = H1ArmController()
        except Exception as e:
            logger.error(f"Master: failed initalizing h1 controllers: {e}")
        
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
        if last_sol_q is None:
            self.h1arm.SetMotorPose(q_poseList, q_tau_ff, True)
            return True

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
        # logger.debug(f"Master: Process ID (PID) {os.getpid()}")
        try:
            while not self.end_event.is_set():
                logger.info("Master: waiting to start")
                self.session_start_event.wait()
                logger.info("Master: start event recvd. clearing start event. starting session")
                self.run_session()
                logger.info("Master: merging data...")
                if not self.failure_event.is_set():
                    self.merge_data() # TODO: maybe a separate thread?
                    logger.info("Master: merge finished. Preparing for a new run...")
                else:
                    self.delete_last_data()
                    logger.info("Master: delete finished. Preparing for a new run...")
                self.reset()
                logger.info("Master: reset finished")
        finally:
            logger.info("Master: finished")

    def get_h1_data(self):
        armstate, armv = self.h1arm.GetMotorState()
        legstate, _ = self.h1arm.GetLegState()
        handstate = self.h1hand.get_hand_state()
        imustate = self.h1arm.GetIMUState()

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
        if "dirname" not in self.shared_data:
            logger.error("Master: failed to get dirname")
            exit(-1)
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
            armstate, armv = self.get_h1_data()
            motor_time = time.time() # TODO: might be late here/ consider puting it before getmotorstate

            get_tv_success, head_rmat, left_pose, right_pose, left_qpos, right_qpos = self.get_teleoperator_data()

            if not get_tv_success:
                continue

            sol_q, tau_ff, ik_flag = self.arm_ik.ik_fun(left_pose, right_pose, armstate, armv)

            ik_time = time.time()

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

    def delete_last_data(self):
        # TODO: auto delete
        with open(self.shared_data["dirname"] + "/failed", "w"):
            pass

class TeleopManager:
    def __init__(self, task_name="default_task", debug=False):
        self.task_name = task_name
        logger.info(f"#### (Task: {self.task_name}):")
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)



        self.manager = Manager()
        self.shared_data = self.manager.dict()

        self.shared_data["kill_event"] = self.manager.Event()
        self.shared_data["session_start_event"] = self.manager.Event()
        self.shared_data["failure_event"] = self.manager.Event()
        self.shared_data["end_event"] = self.manager.Event()# TODO: redundent
        self.kill_event = self.shared_data["kill_event"]
        self.session_start_event = self.shared_data["session_start_event"]
        self.end_event = self.shared_data["end_event"]
        self.failure_event = self.shared_data["failure_event"]

        self.h1_shm = shared_memory.SharedMemory(create=True, size=45 * np.dtype(np.float64).itemsize)
        self.h1_shm_array = np.ndarray((45,), dtype=np.float64, buffer=self.h1_shm.buf)
        self.teleop_shm = shared_memory.SharedMemory(create=True, size=65 * np.dtype(np.float64).itemsize)
        self.teleop_shm_array = np.ndarray((65,), dtype=np.float64, buffer=self.teleop_shm.buf)

        self.shared_data["h1_shm_array"] = self.h1_shm_array
        self.shared_data["teleop_shm_array"] = self.teleop_shm_array

        def run_taskmaster():
            taskmaster = RobotTaskmaster(self.task_name, self.shared_data)
            taskmaster.start()

        def run_dataworker():
            taskworker = RobotDataWorker(self.shared_data)
            taskworker.start()

        self.taskmaster_proc = Process(target=run_taskmaster)
        self.dataworker_proc = Process(target=run_dataworker)

    def start_processes(self):
        logger.info("Starting taskmaster and dataworker processes.")
        self.taskmaster_proc.start()
        self.dataworker_proc.start()

    def update_directory(self):
        dirname = time.strftime(f"demos/{self.task_name}/%Y%m%d_%H%M%S")
        self.shared_data["dirname"] = dirname
        os.makedirs(dirname, exist_ok=True)
        os.makedirs(os.path.join(dirname, "color"), exist_ok=True)
        os.makedirs(os.path.join(dirname, "depth"), exist_ok=True)
        logger.info(f"Data directory set to: {dirname}")

    def start_session(self):
        self.update_directory()
        self.failure_event.clear()
        self.kill_event.clear()
        self.session_start_event.set()
        logger.info("Session started.")

    def stop_session(self):
        self.kill_event.set()
        self.session_start_event.clear()
        logger.info("Session stopped.")

    def cleanup(self):
        logger.info("Cleaning up processes and shared resources...")
        self.end_event.set()
        self.kill_event.set()
        self.session_start_event.set()
        self.manager.shutdown()
        self.taskmaster_proc.terminate()
        self.dataworker_proc.terminate()

        self.taskmaster_proc.kill()
        self.dataworker_proc.kill()

        self.taskmaster_proc.join(timeout=2)
        self.dataworker_proc.join(timeout=2)

        if self.taskmaster_proc.is_alive():
            logger.warning("Forcing termination of taskmaster process.")
            self.taskmaster_proc.kill()
            self.taskmaster_proc.join(timeout=2)

        if self.dataworker_proc.is_alive():
            logger.warning("Forcing termination of dataworker process.")
            self.dataworker_proc.kill()
            self.dataworker_proc.join(timeout=2)

        self.h1_shm.close()
        self.h1_shm.unlink()
        self.teleop_shm.close()
        self.teleop_shm.unlink()
        logger.info("Cleanup complete.")

    def run_command_loop(self):
        last_cmd = None
        finished = 0
        failed = 0
        logger.info("Press 's' to start, 'q' to stop/merge, 'd' for a failure case, 'exit' to quit.")
        try:
            while True:
                user_input = input("> ").lower()
                if user_input == "s" and last_cmd != "s":
                    logger.info(f"Session count - Finished: {finished}, Failed: {failed}")
                    self.start_session()
                    last_cmd = "s"
                elif user_input == "q":
                    self.stop_session()
                    finished += 1
                    last_cmd = "q"
                elif user_input == "d":
                    self.failure_event.set()
                    self.stop_session()
                    failed += 1
                    last_cmd = "d"
                elif user_input == "exit":
                    self.cleanup()
                    sys.exit(0)
                else:
                    logger.info("Invalid command. Use 's' to start, 'q' to stop/merge, 'exit' to quit.")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Exiting...")
            self.cleanup()
            sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Teleoperation Data Collector")
    parser.add_argument("--task_name", type=str, default="default_task", help="Name of the task")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    manager = TeleopManager(task_name=args.task_name, debug=args.debug)
    manager.start_processes()
    manager.run_command_loop()
    # TODO: run in two separate terminals for debuggnig
