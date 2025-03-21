import threading
import time

import numpy as np
from unitree_dds_wrapper.idl import unitree_go
from unitree_dds_wrapper.publisher import Publisher
from unitree_dds_wrapper.subscription import Subscription


class H1HandController:
    def __init__(self):
        self.stop_event = threading.Event()
        self.cmd = unitree_go.msg.dds_.MotorCmds_()
        self.state = unitree_go.msg.dds_.MotorStates_()
        self.labels = {
            "open": np.ones(6),
            "close": np.zeros(6),
            "half": np.full(6, 0.5),
        }
        self.lock = threading.Lock()
        self.init_dds()

    def init_dds(self):
        self.handcmd = Publisher(unitree_go.msg.dds_.MotorCmds_, "rt/inspire/cmd")
        self.handstate = Subscription(
            unitree_go.msg.dds_.MotorStates_, "rt/inspire/state"
        )
        self.cmd.cmds = [unitree_go.msg.dds_.MotorCmd_() for _ in range(12)]
        self.state.states = [unitree_go.msg.dds_.MotorState_() for _ in range(12)]

        self.report_rpy_thread = threading.Thread(target=self.subscribe_state)
        self.report_rpy_thread.start()

    def subscribe_state(self):
        while not self.stop_event.is_set():
            if self.handstate.msg:
                self.state = self.handstate.msg
            time.sleep(0.01)

    # def ctrl(self, label):
    #     if label in self.labels:
    #         self.ctrl(self.labels[label], self.labels[label])
    #     else:
    #         print(f"Invalid label: {label}")

    def ctrl(self, right_angles, left_angles):
        for i in range(6):
            self.cmd.cmds[i].q = right_angles[i]
            self.cmd.cmds[i + 6].q = left_angles[i]
        self.handcmd.msg.cmds = self.cmd.cmds
        self.handcmd.write()

    def get_hand_state(self):
        with self.lock:
            q = np.array([self.state.states[i].q for i in range(12)])
            return q

    def get_right_q(self):
        with self.lock:
            q = np.array([self.state.states[i].q for i in range(6)])
            return q

    def get_left_q(self):
        with self.lock:
            q = np.array([self.state.states[i + 6].q for i in range(6)])
            return q

    def shutdown(self):
        self.stop_event.set()
        self.report_rpy_thread.join()

    def reset(self):
        if self.stop_event.is_set():
            self.stop_event.clear()
        self.report_rpy_thread = threading.Thread(target=self.subscribe_state)
        self.report_rpy_thread.start()
        print("H1HandController has been reset.")
