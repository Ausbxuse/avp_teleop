import os
import sys
import time

import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class Arm_IK:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.robot = pin.RobotWrapper.BuildFromURDF('../assets/h1_2/h1_2.urdf', '../assets/h1_2')
        # self.robot = pin.RobotWrapper.BuildFromURDF('../../assets/h1_2/h1_2.urdf', '../../assets/h1_2/') # for test

        self.mixed_jointsToLockIDs = ["left_hip_yaw_joint",
                                      "left_hip_pitch_joint",
                                      "left_hip_roll_joint",
                                      "left_knee_joint",
                                      "left_ankle_pitch_joint",
                                      "left_ankle_roll_joint",
                                      "right_hip_yaw_joint",
                                      "right_hip_pitch_joint",
                                      "right_hip_roll_joint",
                                      "right_knee_joint",
                                      "right_ankle_pitch_joint",
                                      "right_ankle_roll_joint",
                                      "torso_joint",
                                      "L_index_proximal_joint",
                                      "L_index_intermediate_joint",
                                      "L_middle_proximal_joint",
                                      "L_middle_intermediate_joint",
                                      "L_pinky_proximal_joint",
                                      "L_pinky_intermediate_joint",
                                      "L_ring_proximal_joint",
                                      "L_ring_intermediate_joint",
                                      "L_thumb_proximal_yaw_joint",
                                      "L_thumb_proximal_pitch_joint",
                                      "L_thumb_intermediate_joint",
                                      "L_thumb_distal_joint",
                                      "R_index_proximal_joint",
                                      "R_index_intermediate_joint",
                                      "R_middle_proximal_joint",
                                      "R_middle_intermediate_joint",
                                      "R_pinky_proximal_joint",
                                      "R_pinky_intermediate_joint",
                                      "R_ring_proximal_joint",
                                      "R_ring_intermediate_joint",
                                      "R_thumb_proximal_yaw_joint",
                                      "R_thumb_proximal_pitch_joint",
                                      "R_thumb_intermediate_joint",
                                      "R_thumb_distal_joint"
                                      ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # for i, joint in enumerate(self.reduced_robot.model.joints):
        #     joint_name = self.reduced_robot.model.names[i]
        #     print(f"Joint {i}: {joint_name}, ID: {joint.id}")


        self.reduced_robot.model.addFrame(
            pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.1,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        
        self.reduced_robot.model.addFrame(
            pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.1,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        

        self.init_data = np.zeros(self.reduced_robot.model.nq)




        # Initialize the Meshcat visualizer  for visualization
        #self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        #self.vis.initViewer(open=False) # True---False
        #self.vis.loadViewerModel("pinocchio") 
        #self.vis.displayFrames(False, frame_ids=[113, 114], axis_length = 0.15, axis_width = 5) # True ---False
        #self.vis.display(pin.neutral(self.reduced_robot.model))

        # for i in range(self.reduced_robot.model.nframes):
        #    frame = self.reduced_robot.model.frames[i]
        #    frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #    print(f"Frame ID: {frame_id}, Name: {frame.name}")

        # Enable the display of end effector target frames with short axis lengths and greater width.
        frame_viz_names = ['L_ee_target', 'R_ee_target']
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0],
                      [0, 0, 0], [0, 1, 0],
                      [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        FRAME_AXIS_COLORS = (
            np.array([[1, 0, 0], [1, 0.6, 0],
                      [0, 1, 0], [0.6, 1, 0],
                      [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        )
        axis_length = 0.1
        axis_width = 10
        # for frame_viz_name in frame_viz_names:
        #     self.vis.viewer[frame_viz_name].set_object(
        #         mg.LineSegments(
        #             mg.PointsGeometry(
        #                 position=axis_length * FRAME_AXIS_POSITIONS,
        #                 color=FRAME_AXIS_COLORS,
        #             ),
        #             mg.LineBasicMaterial(
        #                 linewidth=axis_width,
        #                 vertexColors=True,
        #             ),
        #         )
        #      )





        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.L_hand_id].inverse() * cpin.SE3(self.cTf_l)
                    ).vector,
                    cpin.log6(
                        self.cdata.oMf[self.R_hand_id].inverse() * cpin.SE3(self.cTf_r)
                    ).vector
                )
            ],
        )
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # for smooth
        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost  + 0.02 * self.regularization + 0.01 * self.smooth_cost) # for smooth

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-10
            },
            'print_time':False
        }
        self.opti.solver("ipopt", opts)

    def adjust_pose(self, human_left_pose, human_right_pose, human_arm_length=0.55, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def ik_fun(self, left_pose, right_pose, motorstate=None, motorV=None):
        if motorstate is not None:
            self.init_data = motorstate
        # self.init_data = np.array([-1.2645,   0.14744,  0.44173,  0.75885,  1.63288, -0.23551, -0.4231 ,
        # -1.37424, -0.45977,  0.61784, 1.09254, -0.6219,   0.26311,  0.09623])
        # self.init_data = np.array([-1.01593, -0.41146 , 0.40868 , 0.51792,-0.81013 , 0.43169,  0.18019, -1.01593, -0.41146 , 0.40868 , 0.51792,-0.81013 , 0.43169,  0.18019])
        
        self.opti.set_initial(self.var_q, self.init_data)

        left_pose, right_pose = self.adjust_pose(left_pose, right_pose)

        # self.vis.viewer['L_ee_target'].set_transform(left_pose)     # for visualization
        # self.vis.viewer['R_ee_target'].set_transform(right_pose)    # for visualization

        self.opti.set_value(self.param_tf_l, left_pose)
        self.opti.set_value(self.param_tf_r, right_pose)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)
            

            # self.vis.display(sol_q) # for visualization


            self.init_data = sol_q

            if motorV is not None:
                v =motorV * 0.0
            else:
                v = (sol_q-self.init_data ) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q,v,np.zeros(self.reduced_robot.model.nv))

            return sol_q, tau_ff ,True
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return sol_q, '',False
        
if __name__ == "__main__":
    arm_ik = Arm_IK()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.3, +0.2, 0.2]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.3, -0.2, 0.2]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == 's':

        for i in range(150):
            angle = rotation_speed * i
            L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
            R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

            L_tf_target.translation += np.array([0.001,  0.001, 0.001])
            R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            arm_ik.ik_fun(L_tf_target.homogeneous, R_tf_target.homogeneous)
            time.sleep(0.02)
