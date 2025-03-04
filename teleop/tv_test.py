import json
import os
import random
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

# --- Define Mock Classes for the Dependencies ---


class MockArmController:
    def GetMotorState(self):
        # Return dummy motor state and velocity (fast response)
        return np.zeros(35), np.zeros(35)

    def GetLegState(self):
        return np.zeros(35), np.zeros(35)

    def SetMotorPose(self, pose, tau_ff):
        # Execute almost instantly
        return


class MockHandController:
    def __init__(self):
        # Flag to record if the hand control method was called
        self.crtl_called = False

    def get_hand_state(self):
        return np.zeros(10)

    def crtl(self, right_hand_angles, left_hand_angles):
        # Mark that the hand controller's control method was called
        self.crtl_called = True


class MockTeleoperator:
    def __init__(self, config):
        pass

    def step(self):
        # Sleep for a random duration between 0.1 and 0.6 seconds
        time.sleep(random.uniform(0.1, 0.6))
        # Return dummy data: an identity head rotation matrix, dummy poses, and zeros for qpos.
        head_rmat = np.eye(3)
        left_pose = np.eye(4)
        right_pose = np.eye(4)
        left_qpos = np.zeros(10)
        right_qpos = np.zeros(10)
        return head_rmat, left_pose, right_pose, left_qpos, right_qpos


class MockArmIK:
    def ik_fun(self, left_pose, right_pose, armstate, armv):
        # Sleep for a random duration between 0.02 and 0.2 seconds
        time.sleep(random.uniform(0.02, 0.2))
        # Return dummy solutions (for the 14 joints) and a flag indicating success
        sol_q = np.zeros(14)
        tau_ff = np.zeros(14)
        return sol_q, tau_ff, True


class MockLidarProcess:
    def __init__(self, dirname):
        self.dirname = dirname

    def run(self):
        # Do nothing for testing
        return

    def cleanup(self):
        # Do nothing for testing
        return


# Define a dummy robot_data_worker to avoid actual ZMQ or file I/O delays.
def dummy_robot_data_worker(
    dirname, stop_event, start_time, h1arm, h1hand, teleop_lock, teleoperator
):
    # Simulate a minimal worker that just sleeps until stop_event is set.
    while not stop_event.is_set():
        time.sleep(0.1)
    return


# --- Patch the Dependencies in the RobotTaskmaster Module ---
#
# The following patches override the real classes with our mocks:
#   - H1ArmController -> MockArmController
#   - H1HandController -> MockHandController
#   - VuerTeleop      -> MockTeleoperator
#   - Arm_IK         -> MockArmIK
#   - LidarProcess   -> MockLidarProcess
#   - robot_data_worker -> dummy_robot_data_worker
#
@patch("robot_taskmaster.H1ArmController", new=MockArmController)
@patch("robot_taskmaster.H1HandController", new=MockHandController)
@patch("robot_taskmaster.VuerTeleop", new=MockTeleoperator)
@patch("robot_taskmaster.Arm_IK", new=MockArmIK)
@patch("robot_taskmaster.LidarProcess", new=MockLidarProcess)
@patch("robot_taskmaster.robot_data_worker", new=dummy_robot_data_worker)
class TestRobotTaskmaster(unittest.TestCase):
    def setUp(self):
        # Import the RobotTaskmaster class after patching its dependencies.
        from tv import RobotTaskmaster

        self.stop_event = threading.Event()
        # Create a temporary directory so the test doesn't pollute your workspace.
        self.temp_dir = tempfile.TemporaryDirectory()
        # Instantiate RobotTaskmaster; by default it creates directories based on the current time.
        self.taskmaster = RobotTaskmaster("test_task", self.stop_event)
        # Override the demo directory to be within the temporary directory.
        self.taskmaster.dirname = os.path.join(self.temp_dir.name, "test_task")

    def tearDown(self):
        self.taskmaster.stop()
        self.temp_dir.cleanup()

    def test_taskmaster_runs_and_stops(self):
        """
        Test that the RobotTaskmaster can start in a separate thread, run for a short time,
        and then stop properly.
        """
        thread = threading.Thread(target=self.taskmaster.start)
        thread.start()
        # Let the taskmaster run for 2 seconds.
        time.sleep(2)
        self.taskmaster.stop()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive(), "Taskmaster thread should have stopped.")

    def test_safely_set_motor(self):
        """
        Test the safelySetMotor method with dummy inputs to ensure it returns True when
        the IK flag is True.
        """
        sol_q = np.zeros(14)
        tau_ff = np.zeros(14)
        armstate = np.zeros(35)
        right_qpos = np.zeros(10)
        left_qpos = np.zeros(10)
        last_sol_q = None
        result = self.taskmaster.safelySetMotor(
            True, sol_q, last_sol_q, tau_ff, armstate, right_qpos, left_qpos
        )
        self.assertTrue(
            result, "safelySetMotor should return True when IK flag is True."
        )

    def test_hand_controller_usage(self):
        """
        Test that the mocked hand controller is actually used by checking that its crtl method is called.
        """
        # Reset the flag in the hand controller.
        self.taskmaster.h1hand.crtl_called = False
        sol_q = np.zeros(14)
        tau_ff = np.zeros(14)
        armstate = np.zeros(35)
        right_qpos = np.zeros(10)
        left_qpos = np.zeros(10)
        last_sol_q = None
        # Call safelySetMotor which should invoke the hand controller's crtl method.
        self.taskmaster.safelySetMotor(
            True, sol_q, last_sol_q, tau_ff, armstate, right_qpos, left_qpos
        )
        self.assertTrue(
            self.taskmaster.h1hand.crtl_called,
            "The mocked hand controller's crtl method should have been called.",
        )

    def test_merge_data(self):
        """
        Test the merge_data method by creating dummy robot and IK data files along with a
        dummy lidar file. After merging, the output JSON should exist and be valid.
        """
        robot_data_path = os.path.join(self.taskmaster.dirname, "robot_data.txt")
        ik_data_path = os.path.join(self.taskmaster.dirname, "ik_data.json")
        dummy_time = time.time()
        dummy_robot_data = [
            {
                "time": dummy_time,
                "arm_state": [0] * 35,
                "leg_state": [0] * 35,
                "hand_state": [0] * 10,
                "image": "color/frame_000000.jpg",
                "depth": "depth/frame_000000.jpg",
                "imu_omega": [0, 0, 0],
                "imu_rpy": [0, 0, 0],
                "ik_data": None,
                "lidar": None,
            }
        ]
        with open(robot_data_path, "w") as f:
            json.dump(dummy_robot_data, f)
        dummy_ik_data = [
            {
                "right_angles": [0],
                "left_angles": [0],
                "armtime": dummy_time,
                "iktime": dummy_time,
                "sol_q": [0] * 14,
                "tau_ff": [0] * 14,
                "head_rmat": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "left_pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "right_pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        ]
        with open(ik_data_path, "w") as f:
            json.dump(dummy_ik_data, f)
        # Create a dummy lidar file
        lidar_dir = os.path.join(self.taskmaster.dirname, "lidar")
        os.makedirs(lidar_dir, exist_ok=True)
        lidar_file = os.path.join(lidar_dir, f"{dummy_time}.pcd")
        with open(lidar_file, "w") as f:
            f.write("dummy lidar content")
        # Call merge_data and verify that the merged file exists and contains valid JSON.
        self.taskmaster.merge_data()
        merged_file = os.path.join(self.taskmaster.dirname, "merged_data.json")
        self.assertTrue(os.path.exists(merged_file), "Merged data file should exist.")
        with open(merged_file, "r") as f:
            merged_data = json.load(f)
        self.assertIsInstance(merged_data, list, "Merged data should be a list.")


if __name__ == "__main__":
    unittest.main()
