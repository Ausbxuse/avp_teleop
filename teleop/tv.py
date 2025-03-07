import argparse
import gc
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

FREQ = 30
DELAY = 1 / FREQ
CHUNK_SIZE = 100

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class AsyncImageWriter:
    def __init__(self):
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                filename, image = self.queue.get(timeout=0.5)
                cv2.imwrite(filename, image)
            except queue.Empty:
                continue

    def write_image(self, filename, image):
        self.queue.put((filename, image))

    def close(self):
        self.stop_event.set()
        self.thread.join()

class AsyncWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        with open(self.filepath, "a") as f:
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    item = self.queue.get(timeout=0.5)
                    # logger.debug(f"async writer: writing elements {item}")
                    f.write(item + "\n")
                    # Optionally flush after each write:
                    # f.flush()
                except queue.Empty:
                    continue

    def write(self, item):
        self.queue.put(item)

    def close(self):
        self.stop_event.set()
        self.thread.join()

class IKDataWriter:
    def __init__(self, dirname, buffer_size=100):
        self.buffer = []
        self.filepath = os.path.join(dirname, "ik_data.jsonl")
        self.async_writer = AsyncWriter(os.path.join(dirname, "ik_data.jsonl"))
        self.buffer_size = buffer_size  # Buffer size is no longer used here.

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

        logger.info("loading robot and IK data for merging.") # NOTE: stuck here when exit
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
        if self.proc.poll() is None:  # if the process is still running
            logger.info("Sending SIGINT to the lidar process...")
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()  # force kill after timeout
                logger.info("Lidar process killed after timeout.")


class RobotDataWorker:
    def __init__(
        self, dirname_queue, stop_event, start_event, h1_shm_array, teleop_shm_queue
    ):
        self.dirname_queue = dirname_queue
        self.stop_event = stop_event
        self.start_event = start_event
        self.h1_shm_array = h1_shm_array
        self.teleop_shm_queue = teleop_shm_queue
        self.h1_lock = Lock()
        self.frame_idx = 0
        self.teleoperator = VuerTeleop("inspire_hand.yml")
        logger.debug("started vuer")

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect("tcp://192.168.123.162:5556")
        self.socket.setsockopt(zmq.RCVTIMEO, 200)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.async_image_writer = AsyncImageWriter()
        self.robot_data_writer = None

    def _sleep_until_mod33(self, time_curr):
        integer_part = int(time_curr)
        decimal_part = time_curr - integer_part
        ms_part = int(decimal_part * 1000) % 100

        print(time_curr)
        next_ms_part = ((ms_part // 33) + 1) * 33 % 100
        hundred_ms_part = int(decimal_part * 10 % 10)
        if next_ms_part == 32:
            hundred_ms_part += 1
        print(next_ms_part, hundred_ms_part)

        next_capture_time = integer_part + next_ms_part / 1000 + hundred_ms_part / 10
        if (next_capture_time - time_curr) < 0:
            next_capture_time += 1
        time.sleep(next_capture_time - time_curr)

    def _recv_zmq_frame(self):
        compressed_data = b""
        while not self.stop_event.is_set():  # TODO: verify correctness
            chunk = self.socket.recv()
            compressed_data += chunk
            if len(chunk) < 120000:  # Check for last chunk
                break

        try:
            data = zlib.decompress(compressed_data)
            frame_data = pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed decompressing or unpickling frame data: {e}")
            return None, None

        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)  # np: (height, width)

        if frame is None:
            logger.error("Failed to decode frame!")
            return None, None

        color_frame = frame[:, : frame.shape[1] // 2]
        depth_frame = frame[:, frame.shape[1] // 2 :]
        return color_frame, depth_frame

    def teleop_update_thread(self, shm_name):
        shm = shared_memory.SharedMemory(name=shm_name)
        teleop_array = np.ndarray((65,), dtype=np.float64, buffer=shm.buf)
        while not self.stop_event.is_set():
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                self.teleoperator.step()
            )
            # logger.debug(f"teleop step thread: {head_rmat}, {left_pose}, {right_pose}")
            teleop_array[0:9] = head_rmat.flatten()
            teleop_array[9:25] = left_pose.flatten()
            teleop_array[25:41] = right_pose.flatten()
            teleop_array[41:53] = np.array(left_qpos).flatten()
            teleop_array[53:65] = np.array(right_qpos).flatten()
            time.sleep(1.0 / FREQ)
        shm.close()

    def image_buffer_thread(self, image_queue):
        while not self.stop_event.is_set():
            try:
                frame = image_queue.get(timeout=0.1)
                np.copyto(self.teleoperator.img_array, frame)
                logger.debug("image_buf_thread: copied frame")
            except queue.Empty:
                logger.debug("image_buf_thread: empty image")
                continue  # No image available yet, loop again

    def get_robot_data(self, color_frame, depth_frame, time_curr):
        logger.debug(f"worker: starting to get robot data")
        color_filename = os.path.join(
            self.dirname, f"color/frame_{self.frame_idx:06d}.jpg"
        )
        depth_filename = os.path.join(
            self.dirname, f"depth/frame_{self.frame_idx:06d}.jpg"
        )
        if color_frame is not None and depth_frame is not None:
            self.async_image_writer.write_image(color_filename, color_frame)
            self.async_image_writer.write_image(depth_filename, depth_frame)
            # cv2.imwrite(color_filename, color_frame)
            # cv2.imwrite(depth_filename, depth_frame)
            logger.debug(
                f"Saved color frame to {color_filename} and depth frame to {depth_filename}"
            )

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
        logger.debug(f"worker: finish getting robot data")
        return robot_data

    def update_dirname(self, dirname):
        self.dirname = dirname
        os.makedirs(self.dirname, exist_ok=True)
        os.makedirs(os.path.join(self.dirname, "color"), exist_ok=True)
        os.makedirs(os.path.join(self.dirname, "depth"), exist_ok=True)
        logger.info(f"RobotDataWorker: updated dirname to {self.dirname}")

    def start(self):
        try:
            while True:
                self.run_session()
        finally:
            self.context.term()

    def run_session(self):
        try:
            dirname = self.dirname_queue.get()
            self.update_dirname(dirname)
        except queue.Empty:
            logger.error("robot_data_writer: dirname queue empty!")
            pass


        logger.info("RobotDataWorker waiting for new session start (start_event).")
        self.start_event.wait()
        self.start_event.clear()
        logger.info("RobotDataWorker starting new session.")

        self.robot_data_writer = AsyncWriter(
            os.path.join(self.dirname, "robot_data.jsonl")
        )

        teleop_shm = shared_memory.SharedMemory(
            create=True, size=65 * np.dtype(np.float64).itemsize
        )
        self.teleop_shm_queue.put(teleop_shm.name)

        teleop_thread = threading.Thread(
            target=self.teleop_update_thread,
            args=(teleop_shm.name,),
        )
        teleop_thread.daemon = True
        teleop_thread.start()

        logger.info("RobotDataworker: teleop step started")

        image_queue = Queue(maxsize=10)
        image_thread = threading.Thread(
            target=self.image_buffer_thread,
            args=(image_queue,),
        )
        image_thread.daemon = True
        # image_thread.start()

        robot_data_list = []
        # log_filename = os.path.join(self.dirname, "robot_data.jsonl")

        is_first = True

        try:
            while not self.stop_event.is_set():
                color_frame, depth_frame = self._recv_zmq_frame()
                logger.debug("got frame")
                time_curr = time.time()
                if color_frame is not None:
                    resized_frame = cv2.resize(
                        color_frame, (1280, 720), interpolation=cv2.INTER_LINEAR
                    )
                    np.copyto(self.teleoperator.img_array, np.array(resized_frame))
                    # image_queue.put(resized_frame)
                if is_first:
                    is_first = False
                    self._sleep_until_mod33(time.time())
                    initial_capture_time = time.time()
                    logger.debug(f"initial_capture_time is {initial_capture_time}")
                    robot_data = self.get_robot_data(
                        color_frame, depth_frame, time.time()
                    )
                    # robot_data_list.append(robot_data)
                    self.robot_data_writer.write(json.dumps(robot_data))
                    last_robot_data = robot_data
                    self.frame_idx += 1
                    continue

                next_capture_time = initial_capture_time + self.frame_idx * DELAY
                time_curr = time.time()
                logger.debug(
                    f"[worker process] next_capture_time - time_curr: {next_capture_time - time_curr}"
                )

                if time_curr <= next_capture_time:
                    time.sleep(next_capture_time - time_curr)
                    robot_data = self.get_robot_data(
                        color_frame, depth_frame, time.time()
                    )
                    # robot_data_list.append(robot_data)
                    self.robot_data_writer.write(json.dumps(robot_data))
                else:
                    logger.error(
                        "worker process: runner did not finish within 33ms, reusing previous data"
                    )
                    if "last_robot_data" in locals() and last_robot_data is not None:
                        last_robot_data["time"] = time.time()
                        # robot_data_list.append(last_robot_data)
                        self.robot_data_writer.write(json.dumps(robot_data))
                    else:
                        logger.error(
                            "worker process: no previous data available, generating null data"
                        )
                        robot_data = self.get_robot_data(None, None, time.time())
                        robot_data_list.append(robot_data)
                last_robot_data = robot_data
                self.frame_idx += 1

        except KeyboardInterrupt as e:
            logger.info(f"[INTR] keyboard interrupted: {e}")
        except Exception as e:
            logger.error(f"[ERROR] robot_data_worker encountered an error: {e}")

        finally:
            # if robot_data_list:
            #     with open(log_filename, "a") as f:
            #         for data in robot_data_list:
            #             json.dump(data, f)
            #             f.write("\n")
            #     logger.debug(f"Flushed remaining robot data to {log_filename}")
            # # self.context.term()
            logger.info("robot_data_worker process has exited.")
            self.stop_event.set()
            teleop_thread.join()
            image_thread.join()
            self.robot_data_writer.close()
            # teleop_shm.close()
            # teleop_shm.unlink()
            self.reset()

    def reset(self):
        # TODO: finish rest
        self.frame_idx = 0


# Teleop and datacollector
class RobotTaskmaster:
    def __init__(self, task_name):
        self.task_name = task_name
        self.stop_event = Event()
        self.start_event = Event()
        self.dirname_queue = Queue()
        self.teleop_lock = Lock()
        self.h1hand = H1HandController()
        self.h1arm = H1ArmController()
        self.arm_ik = Arm_IK()
        self.first = True
        self.lidar_proc = None
        self.ik_writer = None
        self.running = False
        self.teleop_shm_queue = Queue()
        self.h1_shm_queue = Queue()
        self.dirname = time.strftime(f"demos/{self.task_name}/%Y%m%d_%H%M%S")
        self.dirname_queue.put(self.dirname)
        os.makedirs(self.dirname)
        os.makedirs(os.path.join(self.dirname, "color"))
        os.makedirs(os.path.join(self.dirname, "depth"))
        self.h1_shm = shared_memory.SharedMemory(
            create=True, size=45 * np.dtype(np.float64).itemsize
        )
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
            + [np.pi ] * 2  # left wrists
            + [np.pi / 3] * 5
            + [np.pi ] * 2
        )
        if last_sol_q is not None and np.any(
            np.abs(last_sol_q - sol_q) > dynamic_thresholds
        ):
            logger.error("ik movement too large!")
            return False
        if not ik_flag:
            q_poseList[13:27] = armstate
            q_tau_ff = np.zeros(35)
            logger.error("ik flag false!")
            return False

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
        self.lidar_proc = LidarProcess(self.dirname)
        self.lidar_proc.run()
        self.start_event.set()
        self.running = True
        self.ik_writer = IKDataWriter(self.dirname)

        teleop_shm_name = self.teleop_shm_queue.get()
        self.teleop_shm = shared_memory.SharedMemory(name=teleop_shm_name)
        self.teleop_shm_array = np.ndarray(
            (65,), dtype=np.float64, buffer=self.teleop_shm.buf
        )

        right_hand_angles = None
        left_hand_angles = None
        last_sol_q = None
        while not self.stop_event.is_set():
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

            logger.debug("robot_master: loop start")
            motor_time = time.time()
            with self.teleop_lock:
                teleop_data = self.teleop_shm_array.copy()
            if np.all(teleop_data == 0):
                logger.debug(f"robot_master: not receving data yet: {teleop_data}")
                continue
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
        if self.lidar_proc is not None:
            self.lidar_proc.cleanup()
        if self.ik_writer is not None:
            self.ik_writer.close()
        self.h1arm.shutdown()
        self.h1hand.shutdown()

        # if self.h1_shm is not None:
        #     try:
        #         self.h1_shm.close()
        #     except Exception as e:
        #         logger.error(f"Error cleaning up h1_shm: {e}")

        # if hasattr(self, "teleop_shm") and self.teleop_shm is not None:
        #     try:
        #         self.teleop_shm.close()
        #         self.teleop_shm.unlink()
        #     except Exception as e:
        #         logger.error(f"Error cleaning up teleop_shm: {e}")

        logger.info("Stopping all threads ended!")

    def reset(self):
        logger.info("Resetting RobotTaskmaster...")
        if self.running:
            self.stop()
        self.stop_event.clear()  # TODO: create a new one?

        self.h1hand.reset()
        self.h1arm.reset()
        self.first = True
        self.running = False

        # try:
        #     self.h1_shm.close()
        #     self.h1_shm.unlink()
        # except Exception as e:
        #     logger.error(f"Error cleaning up h1_shm: {e}")

        self.h1_shm_array[:] = 0

        self.dirname = time.strftime(f"demos/{self.task_name}/%Y%m%d_%H%M%S")
        self.dirname_queue.put(self.dirname)
        os.makedirs(self.dirname, exist_ok=True)
        os.makedirs(os.path.join(self.dirname, "color"), exist_ok=True)
        os.makedirs(os.path.join(self.dirname, "depth"), exist_ok=True)

        self.ik_writer = IKDataWriter(self.dirname)

        logger.info("RobotTaskmaster has been reset and is ready to start again.")

    def merge_data(self):
        merger = DataMerger(self.dirname)
        merger.merge_json()


if __name__ == "__main__":

    # dirname_queue = Queue()
    # stop_event = Event()
    # start_event = Event()
    # dirname_queue.put("heeehee")
    # def run_dataworker(
    #     dirname_queue, stop_event, start_event, h1_shm_array, teleop_shm_queue
    # ):
    #     taskworker = RobotDataWorker(
    #         dirname_queue, stop_event, start_event, h1_shm_array, teleop_shm_queue
    #     )
    #     print("starting")
    #     taskworker.start()
    #
    # teleop_shm_queue = Queue()
    # h1_shm_queue = Queue()
    #
    # h1_shm = shared_memory.SharedMemory(
    #     create=True, size=45 * np.dtype(np.float64).itemsize
    # )
    # h1_shm_array = np.ndarray((45,), dtype=np.float64, buffer=h1_shm.buf)
    #
    # proc =  Process(
    #     target=run_dataworker,
    #     args=(
    #         dirname_queue,
    #         stop_event,
    #         start_event,
    #         h1_shm_array,
    #         teleop_shm_queue,
    #     ),
    # )
    # proc.start()
    # start_event.set()

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

    logger.info(f"Robo master control (Task: {args.task_name}):")
    logger.info("  Press 's' to start the taskmaster")
    logger.info("  Press 'q' to stop and merge data")

    # Although a teleoperator is passed here for API compatibility, it is now unused in the main process.
    task_thread = None
    taskmaster = RobotTaskmaster(args.task_name)

    def run_taskmaster():
        taskmaster.start()

    def run_dataworker(
        dirname_queue, stop_event, start_event, h1_shm_array, teleop_shm_queue
    ):
        taskworker = RobotDataWorker(
            dirname_queue, stop_event, start_event, h1_shm_array, teleop_shm_queue
        )
        taskworker.start()

    def create_robot_data_proc(taskmaster):
        return Process(
            target=run_dataworker,
            args=(
                taskmaster.dirname_queue,
                taskmaster.stop_event,
                taskmaster.start_event,
                taskmaster.h1_shm_array,
                taskmaster.teleop_shm_queue,
            ),
        )

    robot_data_proc = create_robot_data_proc(taskmaster)
    robot_data_proc.start()
    # TODO: fix inconsistent arm time (not strictly 33hz)

    taskmaster_proc = Process(target=taskmaster.start)
    try:
        while True:
            if sys.stdin.closed:  # TODO: why???
                logger.error("Standard input is closed. Continuing...")
                sys.stdin = open("/dev/tty")
                continue
            user_input = input("> ").lower()
            # user_input = await asyncio.to_thread(input, "> ")

            if user_input == "s" and not taskmaster.running:
                # task_thread = threading.Thread(target=run_taskmaster)
                # task_thread.daemon = True
                taskmaster_proc.start()
                logger.info("Started taskmaster and dataworker")

            elif user_input == "q" and taskmaster.running:
                taskmaster.stop()
                taskmaster_proc.join(timeout=1)
                logger.info("Stopping taskmaster")
                if robot_data_proc is not None and robot_data_proc.is_alive():
                    robot_data_proc.join(timeout=5)
                logger.info("Merging data...")
                taskmaster.merge_data()
                logger.info("Merge finished. Preparing for a new run...")
                # taskmaster = RobotTaskmaster(args.task_name)  # TODO: reset instead
                taskmaster.reset()
                logger.info("Ready to rerun!")

            elif user_input == "exit":
                logger.info("Exiting...")
                if taskmaster.running:
                    taskmaster.stop()
                    taskmaster_proc.join(timeout=1)
                logger.debug("Terminating data proc...")
                robot_data_proc.terminate()
                if robot_data_proc is not None and robot_data_proc.is_alive():
                    robot_data_proc.join(timeout=5)
                logger.debug("Data proc terminated")
                gc.collect()
                sys.exit(0)

            else:
                logger.info(
                    "Invalid. Use 's' to start, 'q' to stop/merge, 'exit' to quit."
                )

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Exiting...")
        if taskmaster.running and task_thread is not None:
            taskmaster.stop()
            taskmaster.merge_data()
            taskmaster_proc.join(timeout=1)
    finally:
        sys.exit(0)
