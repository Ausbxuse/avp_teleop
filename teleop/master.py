import os
import sys
import time
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory
from operator import le
from turtle import left

import numpy as np

from lidar import LidarProcess
from merger import DataMerger
from robot_control.robot_arm import H1ArmController
from robot_control.robot_arm_ik import Arm_IK
from robot_control.robot_hand import H1HandController
from utils.logger import logger
from writers import IKDataWriter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class RobotTaskmaster:
    def __init__(self, task_name, shared_data, h1_shm_array, teleop_shm_array):
        self.task_name = task_name

        self.shared_data = shared_data
        self.kill_event = shared_data["kill_event"]
        self.session_start_event = shared_data["session_start_event"]
        self.failure_event = shared_data["failure_event"]  # TODO: redundent
        self.end_event = shared_data["end_event"]  # TODO: redundent

        self.h1_shm_array = h1_shm_array
        self.teleop_shm_array = teleop_shm_array

        self.teleop_lock = Lock()
        try:
            self.h1hand = H1HandController()
            self.h1arm = H1ArmController()
        except Exception as e:
            logger.error(f"Master: failed initalizing h1 controllers: {e}")
            logger.error(f"Master: exiting")
            exit(-1)

        self.arm_ik = Arm_IK()
        self.first = True
        self.lidar_proc = None
        self.ik_writer = None
        self.running = False
        self.h1_lock = Lock()

    def safelySetMotor(self, ik_flag, sol_q, last_sol_q, tau_ff, armstate):
        q_poseList = np.zeros(35)
        q_tau_ff = np.zeros(35)
        q_poseList[13:27] = sol_q
        q_tau_ff[13:27] = tau_ff  # WARN: untested!
        dynamic_thresholds = np.array(
            [np.pi / 3] * 5  # left shoulder and elbow
            + [np.pi] * 2  # left wrists
            + [np.pi / 3] * 5
            + [np.pi] * 2
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
        return True

    def setHandMotors(self, right_qpos, left_qpos):
        if right_qpos is not None and left_qpos is not None:
            right_hand_angles = [1.7 - right_qpos[i] for i in [4, 6, 2, 0]]
            right_hand_angles.append(1.2 - right_qpos[8])
            right_hand_angles.append(0.5 - right_qpos[9])

            left_hand_angles = [1.7 - left_qpos[i] for i in [4, 6, 2, 0]]
            left_hand_angles.append(1.2 - left_qpos[8])
            left_hand_angles.append(0.5 - left_qpos[9])
            self.h1hand.ctrl(right_hand_angles, left_hand_angles)
        return right_hand_angles, left_hand_angles

    def start(self):
        # logger.debug(f"Master: Process ID (PID) {os.getpid()}")
        try:
            while not self.end_event.is_set():
                logger.info("Master: waiting to start")
                self.session_start_event.wait()
                logger.info(
                    "Master: start event recvd. clearing start event. starting session"
                )
                self.run_session()
                logger.info("Master: merging data...")
                if not self.failure_event.is_set():
                    self.merge_data()  # TODO: maybe a separate thread?
                    logger.info("Master: merge finished. Preparing for a new run...")
                else:
                    # self.delete_last_data()
                    logger.info(
                        "Master: not merging. Preparing for a new run to override..."
                    )
                self.reset()
                logger.info("Master: reset finished")
        finally:
            self.stop()
            logger.info("Master: exited")

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
            motor_time = (
                time.time()
            )  # TODO: might be late here/ consider puting it before getmotorstate

            get_tv_success, head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                self.get_teleoperator_data()
            )
            logger.debug("Master: got teleop ddata")

            if not get_tv_success:
                continue

            sol_q, tau_ff, ik_flag = self.arm_ik.ik_fun(
                left_pose, right_pose, armstate, armv
            )

            ik_time = time.time()

            logger.debug(f"Master: moving motor {sol_q}")
            # if self.safelySetMotor(ik_flag, sol_q, last_sol_q, tau_ff, armstate):
            #     last_sol_q = sol_q
            # else:
            #     continue
            #
            # self.setHandMotors(right_qpos, left_qpos)

            logger.debug("Master: writing data")
            logger.debug(f"Master: head_rmat: {head_rmat}")
            self.ik_writer.write_data(
                right_qpos,
                left_qpos,
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
