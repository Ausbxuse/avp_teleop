import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def current_time():
    """Return the current time as a formatted string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# Dummy teleoperator simulating a slow teleop.step() call.
class DummyTeleoperator:
    def step(self):
        print(f"[{current_time()}] [Teleop] Starting teleop step...")
        time.sleep(0.5)  # Simulate delay in teleop computation
        # Return dummy values (head_rmat, left_pose, right_pose, left_qpos, right_qpos)
        result = (
            "head_rmat_value",
            "left_pose_value",
            "right_pose_value",
            "left_qpos_value",
            "right_qpos_value",
        )
        print(f"[{current_time()}] [Teleop] Teleop step completed.")
        return result


# Dummy IK function simulating a slow IK computation.
def dummy_ik_fun(left_pose, right_pose, armstate, armv):
    print(
        f"[{current_time()}] [IK] Starting IK computation with teleop values: left_pose={left_pose}, right_pose={right_pose}"
    )
    time.sleep(0.4)  # Simulate delay in IK computation
    sol_q = "sol_q_value"
    tau_ff = "tau_ff_value"
    ik_flag = True
    print(f"[{current_time()}] [IK] IK computation completed.")
    return sol_q, tau_ff, ik_flag


# Taskmaster that offloads IK computation and updates teleop asynchronously.
class RobotTaskmaster:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.teleoperator = DummyTeleoperator()
        self.armstate = "armstate_value"  # Dummy arm state value
        self.armv = "armv_value"  # Dummy arm velocity value

        # Shared variable to hold the latest teleop result, protected by a lock.
        self._teleop_result = None
        self._teleop_result_lock = threading.Lock()

        # Thread to continuously update teleop result.
        self._teleop_thread = threading.Thread(target=self._update_teleop)
        self._teleop_thread.daemon = True

        # Executor to offload slow IK computations.
        self._ik_executor = ThreadPoolExecutor(max_workers=1)

    def _update_teleop(self):
        """Continuously call teleoperator.step() and update the shared teleop result."""
        while not self.stop_event.is_set():
            result = self.teleoperator.step()
            with self._teleop_result_lock:
                self._teleop_result = result
            # Brief pause before next update
            time.sleep(0.05)

    def start(self):
        print(
            f"[{current_time()}] [Taskmaster] Starting teleop update thread and main loop..."
        )
        self._teleop_thread.start()

        iteration = 0
        # Run for a few iterations for this test.
        while not self.stop_event.is_set() and iteration < 5:
            with self._teleop_result_lock:
                teleop_result = self._teleop_result

            if teleop_result is None:
                print(f"[{current_time()}] [Main Loop] Waiting for teleop result...")
                time.sleep(0.1)
                continue

            print(
                f"[{current_time()}] [Main Loop] Iteration {iteration}: Retrieved teleop result: {teleop_result}"
            )

            # Offload the IK computation using the latest teleop result.
            future = self._ik_executor.submit(
                dummy_ik_fun,
                teleop_result[1],
                teleop_result[2],
                self.armstate,
                self.armv,
            )
            sol_q, tau_ff, ik_flag = future.result()  # Wait for IK result

            print(
                f"[{current_time()}] [Main Loop] Iteration {iteration}: Received IK result: sol_q={sol_q}, tau_ff={tau_ff}, ik_flag={ik_flag}"
            )
            iteration += 1

        print(f"[{current_time()}] [Taskmaster] Main loop finished.")

    def stop(self):
        print(f"[{current_time()}] [Taskmaster] Stopping...")
        self.stop_event.set()
        self._ik_executor.shutdown(wait=True)
        self._teleop_thread.join()
        print(f"[{current_time()}] [Taskmaster] Stopped.")


if __name__ == "__main__":
    stop_event = threading.Event()
    taskmaster = RobotTaskmaster(stop_event)
    try:
        taskmaster.start()
    except KeyboardInterrupt:
        print(f"[{current_time()}] KeyboardInterrupt caught!")
    finally:
        taskmaster.stop()
