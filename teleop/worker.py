import json
import lzma
import os
import pickle
import queue
import sys
import threading
import time
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import zmq

from utilities import VuerTeleop
from utils.logger import logger
from writers import AsyncImageWriter, AsyncWriter

FREQ = 30
DELAY = 1 / FREQ

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class RobotDataWorker:
    def __init__(self, shared_data, h1_shm_array, teleop_shm_array):
        self.shared_data = shared_data
        self.kill_event = shared_data["kill_event"]
        self.session_start_event = shared_data["session_start_event"]
        self.end_event = shared_data["end_event"]  # TODO: redundent
        self.h1_lock = Lock()
        self.teleoperator = VuerTeleop("inspire_hand.yml")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://192.168.123.162:5556")

        self.h1_shm_array = h1_shm_array
        self.teleop_shm_array = teleop_shm_array
        self.depth_kill_event = Event()

        self.depth_queue = Queue()
        self.async_image_writer = AsyncImageWriter()

        self.depth_proc = Process(
            target=self.depth_writer_process,
            args=(self.depth_queue, self.depth_kill_event),
        )

        # resetable vars
        self.frame_idx = 0
        self.last_robot_data = None
        self.robot_data_writer = None

    def depth_writer_process(self, depth_queue, kill_event):
        while not kill_event.is_set():
            try:
                filename, depth_array = depth_queue.get(timeout=0.5)
                # buffer = io.BytesIO()
                # np.save(buffer, depth_array)
                # depth_bytes = buffer.getvalue()
                compressed_data = lzma.compress(depth_array.tobytes(), preset=0)
                with open(filename, "wb") as f:
                    f.write(compressed_data)
            except queue.Empty:
                continue

    def dump_state(self, filename=None):
        """Dump current system state for debugging"""
        if filename is None:
            filename = f"debug_dump_{time.strftime('%Y%m%d_%H%M%S')}.pkl"

        state = {
            "h1_data": self.h1_shm_array.copy(),
            "teleop_data": self.teleop_shm_array.copy(),
            "frame_idx": self.frame_idx if hasattr(self, "frame_idx") else None,
            "timestamp": time.time(),
        }

        with open(filename, "wb") as f:
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
        self.socket.send(b"get_frame")

        message_parts = (
            self.socket.recv_multipart()
        )  # FIXME: BLOCKING might lead to resource leak

        if len(message_parts) != 2 or not message_parts[0] or not message_parts[1]:
            logger.error("Failed to receive complete frame data!")
            return None, None, None, None

        rgb_ir_bytes = message_parts[0]
        np_arr = np.frombuffer(rgb_ir_bytes, np.uint8)
        combined_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if combined_frame is None:
            logger.error("Failed to decode RGB+IR frame!")
            return None, None, None, None

        width_each = combined_frame.shape[1] // 3
        color_frame = combined_frame[:, 0:width_each]
        ir_left_frame = combined_frame[:, width_each : 2 * width_each]
        ir_right_frame = combined_frame[:, 2 * width_each :]

        depth_bytes = message_parts[1]
        height = color_frame.shape[0]
        width = color_frame.shape[1]

        depth_array = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(height, width)

        return color_frame, depth_array, ir_left_frame, ir_right_frame

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
            "image": f"color/frame_{self.frame_idx:06d}.jpg",
            "depth": f"depth/frame_{self.frame_idx:06d}.npy.lzma",
            "imu_omega": imustate[0:3].tolist(),
            "imu_rpy": imustate[3:6].tolist(),
            "ik_data": None,
            "lidar": None,
        }
        # logger.debug(f"worker: finish getting robot data")
        return robot_data

    def start(self):
        # logger.debug(f"Worker: Process ID (PID) {os.getpid()}")
        self.depth_proc.start()
        try:
            while not self.end_event.is_set():
                logger.info(
                    "Worker: waiting for new session start (session_start_event)."
                )
                self.session_start_event.wait()
                logger.info("Worker: starting new session.")
                self.run_session()
                self.async_image_writer.close()
                self.async_image_writer = AsyncImageWriter()
        finally:
            self.socket.close()
            self.context.term()
            self.teleoperator.shutdown()
            self.depth_kill_event.set()
            self.depth_proc.join()
            logger.info("Worker: exited")

    def _write_image_data(self, color_frame, depth_frame):
        logger.debug("Worker: writing robot data")

        color_filename = os.path.join(
            self.shared_data["dirname"], f"color/frame_{self.frame_idx:06d}.jpg"
        )
        depth_filename = os.path.join(
            self.shared_data["dirname"], f"depth/frame_{self.frame_idx:06d}.npy.lzma"
        )

        if color_frame is not None and depth_frame is not None:
            self.async_image_writer.write_image(color_filename, color_frame)
            # compressed_data = lzma.compress(depth_frame.tobytes(), preset=0)
            #
            self.depth_queue.put((depth_filename, depth_frame))
            # np.save(depth_filename, depth_frame)
            logger.debug(
                f"Saved color frame to {color_filename} and depth frame to {depth_filename}"
            )

        else:
            logger.error(f"failed to save image {self.frame_idx}")

    def _write_robot_data(self, color_frame, depth_frame, reuse=False):
        logger.debug(f"Worker: writing robot data")
        self._write_image_data(color_frame, depth_frame)

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
        logger.debug("Worker: initing robot_data_writer")

        self.teleop_thread = threading.Thread(
            target=self.teleop_update_thread, daemon=True
        )
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

        if next_capture_time - time_curr >= 0:
            time.sleep(next_capture_time - time_curr)
            self._write_robot_data(color_frame, depth_frame)
        else:
            logger.warning(
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
                # self.robot_data_writer.close()
                # self.robot_data_writer = AsyncWriter(
                #     os.path.join(self.shared_data["dirname"], "robot_data.jsonl")
                # )
                logger.debug("Worker: initing robot_data_writer")

        except Exception as e:
            logger.error(f"robot_data_worker encountered an error: {e}")

        finally:
            logger.info("Worker begin exiting.")
            # TODO: flush the buffer?
            self.teleop_thread.join(1)
            logger.info("Worker: teleop thread joined.")
            self.robot_data_writer.close()
            logger.info("Worker: writer closed.")
            self.reset()
            logger.info("Worker: closing async image writer.")
            if hasattr(self, "async_image_writer"):
                self.async_image_writer.close()
                logger.info("Worker: async image writer closed.")

            logger.info("Worker process has exited.")

    def reset(self):
        self.frame_idx = 0
        self.initial_capture_time = None
