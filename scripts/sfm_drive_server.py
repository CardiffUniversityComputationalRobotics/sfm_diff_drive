#!/usr/bin/env python3

import rospy

from pedsim_msgs.msg import AgentStates, AgentGroups
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from tf import TransformListener
import tf
import numpy as np
import math
import actionlib
from sfm_diff_drive.msg import (
    SFMDriveFeedback,
    SFMDriveResult,
    SFMDriveAction,
)
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from simple_pid import PID
from actionlib_msgs.msg import GoalID


class SocialForceModelDriveAction(object):

    _feedback = SFMDriveFeedback()
    _result = SFMDriveResult()

    def __init__(self):

        # base variables
        self.goal_set = False

        self.xy_tolerance = 1

        self._action_name = "sfm_drive_node"

        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction
        )

        self.agents_states_register = []
        self.agents_groups_register = []
        self.current_waypoint = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_position = np.array([0, 0, 0], np.dtype("float64"))

        self.robot_max_vel = 0.5
        self.robot_current_vel = np.array([0, 0, 0], np.dtype("float64"))
        self.relaxation_time = 0.5
        self.laser_ranges = np.zeros(360)

        self.agent_radius = 1
        self.force_sigma_obstacle = 0.8

        # for social force computing

        self.lambda_importance = 2
        self.gamma = 0.35
        self.n = 2
        self.n_prime = 3

        # constants for forces
        self.force_factor_desired = 1.0
        self.force_factor_social = 2.1
        self.force_factor_obstacle = 10

        self.tf = TransformListener()

        # PID for rotation
        self.pid_rotation = PID(0.25, 0.1, 0.0001, setpoint=0)
        self.pid_rotation.output_limits = (-0.75, 0.75)

        self._as = actionlib.SimpleActionServer(
            self._action_name,
            SFMDriveAction,
            execute_cb=self.execute_cb,
            auto_start=False,
        )
        self._as.start()

        #! subscribers
        self.agents_states_subs = rospy.Subscriber(
            "/pedsim_simulator/simulated_agents",
            AgentStates,
            self.agents_state_callback,
        )

        self.agents_groups_subs = rospy.Subscriber(
            "/pedsim_simulator/simulated_groups",
            AgentGroups,
            self.agents_groups_callback,
        )
        self.robot_pos_subs = rospy.Subscriber(
            "/pedsim_simulator/robot_position",
            Odometry,
            self.robot_pos_callback,
        )

        self.laser_scan_subs = rospy.Subscriber(
            "/scan_filtered", LaserScan, self.laser_scan_callback
        )

        #! publishers
        self.velocity_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    def check_goal_reached(self):
        if (
            abs(
                np.linalg.norm(
                    np.array(
                        [self.current_waypoint[0], self.current_waypoint[1]],
                        np.dtype("float64"),
                    )
                    - np.array(
                        [self.robot_position[0], self.robot_position[1]],
                        np.dtype("float64"),
                    )
                )
            )
            <= 0.5
        ):
            return True
        return False

    # * callbacks

    def execute_cb(self, goal):
        rospy.loginfo("Starting social drive")
        r_sleep = rospy.Rate(30)
        cancel_move_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        cancel_msg = GoalID()
        cancel_move_pub.publish(cancel_msg)

        self.goal_set = True
        self.current_waypoint = np.array(
            [goal.goal.x, goal.goal.y, goal.goal.z], np.dtype("float64")
        )

        while not self.check_goal_reached():
            complete_force = (
                self.force_factor_desired * self.desired_force()
                + self.force_factor_obstacle * self.obstacle_force()
                + self.force_factor_social * self.social_force()
            )

            print("complete force:", complete_force)

            self.robot_current_vel = self.robot_current_vel + (complete_force / 25)
            print("robot current vel:", self.robot_current_vel)

            speed = np.linalg.norm(self.robot_current_vel)

            if speed > self.robot_max_vel:
                self.robot_current_vel = (
                    self.robot_current_vel
                    / np.linalg.norm(self.robot_current_vel)
                    * self.robot_max_vel
                )

            t = self.tf.getLatestCommonTime("/base_footprint", "/odom")
            position, quaternion = self.tf.lookupTransform(
                "/base_footprint", "/odom", t
            )
            robot_offset_angle = tf.transformations.euler_from_quaternion(quaternion)[2]

            print("offset_angle:", math.degrees(robot_offset_angle))

            angulo_velocidad = angle(
                self.robot_current_vel, np.array([1, 0, 0], np.dtype("float64"))
            )

            print("angulo_velocidad:", math.degrees(angulo_velocidad))

            if self.robot_position[1] > self.current_waypoint[1]:
                angulo_velocidad = angulo_velocidad - robot_offset_angle
            else:
                angulo_velocidad = angulo_velocidad + robot_offset_angle

            print("angulo final (degree):", math.degrees(angulo_velocidad))

            vx = np.linalg.norm(self.robot_current_vel) * math.cos(angulo_velocidad)

            # w = np.linalg.norm(self.robot_current_vel) * math.sin(angulo_velocidad)

            w = self.pid_rotation(angulo_velocidad)

            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = vx
            cmd_vel_msg.angular.z = -w

            self.velocity_pub.publish(cmd_vel_msg)

            print("v lineal:", vx)
            print("w:", w)
            print("#####")
            self._feedback.feedback = "robot moving"
            rospy.loginfo("robot_ moving")
            self._as.publish_feedback(self._feedback)
            r_sleep.sleep()
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0
        cmd_vel_msg.angular.z = 0

        self.velocity_pub.publish(cmd_vel_msg)
        self._result = "waypoint reached"
        rospy.loginfo("waypoint reached")
        self._as.set_succeeded(self._result)

    """
    call back para agarrar los datos del laser
    """

    def laser_scan_callback(self, data):
        self.laser_ranges = data.ranges

    """
    callback para agarrar datos de posicion del robot
    """

    def robot_pos_callback(self, data):
        data_position = data.pose.pose.position
        self.robot_position = np.array(
            [data_position.x, data_position.y, data_position.z], np.dtype("float64")
        )

    """
    callback para obtener lista de info de agentes
    """

    def agents_state_callback(self, data):
        self.agents_states_register = data.agent_states

    """
    callback para obtener los datos de grupos de agentes
    """

    def agents_groups_callback(self, data):
        self.agents_groups_register = data

        # * force functions
        """
        funcion para obtener la fuerza al waypoint
        """

    def desired_force(self):
        desired_direction = self.current_waypoint - self.robot_position
        desired_direction_vec_norm = np.linalg.norm(desired_direction)
        if desired_direction_vec_norm != 0:
            norm_desired_direction = desired_direction / desired_direction_vec_norm
        else:
            norm_desired_direction = np.array([0, 0, 0], np.dtype("float64"))
        desired_force = (
            norm_desired_direction * self.robot_max_vel - self.robot_current_vel
        ) / self.relaxation_time
        print("desired force:", desired_force)
        return desired_force

    """
    funcion para obtener la fuerza de el obstaculo mas cercano
    """

    def obstacle_force(self):
        diff_robot_laser = []
        # obtener valores de el laser sus distancias
        for i in range(0, 360):
            distance = math.sqrt(
                math.pow(self.laser_ranges[i] * math.cos(math.radians(i - 90)), 2)
                + math.pow(self.laser_ranges[i] * math.sin(math.radians(i - 90)), 2)
            )
            diff_robot_laser.append(distance)

        diff_robot_laser = np.array(diff_robot_laser, np.dtype("float64"))

        for i in range(0, 360):
            if diff_robot_laser[i] == np.nan:
                diff_robot_laser[i] = np.inf

        min_index = 0
        tmp_val = 1000
        for i in range(0, 360):
            if diff_robot_laser[i] < tmp_val and diff_robot_laser[i] != 0:
                tmp_val = diff_robot_laser[i]
                min_index = i

        if diff_robot_laser[min_index] < 1:
            print("minimo:", diff_robot_laser[min_index])
            # obtengo la posicion del valor minimo del laser en distancia
            laser_pos = -1 * np.array(
                [
                    self.laser_ranges[min_index]
                    * math.cos(math.radians(min_index - 180)),
                    self.laser_ranges[min_index]
                    * math.sin(math.radians(min_index - 180)),
                    0,
                ],
                np.dtype("float64"),
            )
            # print("laser_pos:", laser_pos)

            laser_vec_norm = np.linalg.norm(laser_pos)
            if laser_vec_norm != 0:
                norm_laser_direction = laser_pos / laser_vec_norm
            else:
                norm_laser_direction = np.array([0, 0, 0], np.dtype("float64"))

            distance = diff_robot_laser[min_index] - self.agent_radius
            force_amount = math.exp(-distance / self.force_sigma_obstacle)
            final_rep_force = force_amount * norm_laser_direction
            print("Obstacle force:", final_rep_force)
            return final_rep_force
        else:
            return np.array([0, 0, 0], np.dtype("float64"))

    """
    funcion para obtener la fuerzas sociales de los alrededores
    """

    def social_force(self):

        force = np.array([0, 0, 0], np.dtype("float64"))

        for i in self.agents_states_register:
            diff_position = (
                np.array(
                    [
                        i.pose.position.x,
                        i.pose.position.y,
                        i.pose.position.z,
                    ],
                    np.dtype("float64"),
                )
                - self.robot_position
            )

            diff_direction = diff_position / np.linalg.norm(diff_position)

            agent_velocity = i.twist.linear
            diff_vel = self.robot_current_vel - np.array(
                [
                    agent_velocity.x,
                    agent_velocity.y,
                    agent_velocity.z,
                ],
                np.dtype("float64"),
            )

            interaction_vector = self.lambda_importance * diff_vel + diff_direction

            interaction_length = np.linalg.norm(interaction_vector)

            interaction_direction = interaction_vector / interaction_length

            theta = angle(interaction_direction, diff_direction)

            B = self.gamma * interaction_length

            force_velocity_amount = -math.exp(
                -np.linalg.norm(diff_position) / B
                - (self.n_prime * B * theta) * (self.n_prime * B * theta)
            )

            force_angle_amount = -number_sign(theta) * math.exp(
                -np.linalg.norm(diff_position) / B
                - (self.n * B * theta) * (self.n * B * theta)
            )

            force_velocity = force_velocity_amount * interaction_direction

            force_angle = force_angle_amount * np.array(
                [
                    -interaction_direction[1],
                    interaction_direction[0],
                    0,
                ],
                np.dtype("float64"),
            )

            force += force_velocity + force_angle
            print("Social force:", force)
        return force

    """
    funcion para correr parte del programa
    """

    def run(self):
        while not rospy.is_shutdown():
            if self.goal_set:
                complete_force = (
                    self.force_factor_desired * self.desired_force()
                    + self.force_factor_obstacle * self.obstacle_force()
                    + self.force_factor_social * self.social_force()
                )

                print("complete force:", complete_force)

                self.robot_current_vel = self.robot_current_vel + (complete_force / 25)
                print("robot current vel:", self.robot_current_vel)

                speed = np.linalg.norm(self.robot_current_vel)

                if speed > self.robot_max_vel:
                    self.robot_current_vel = (
                        self.robot_current_vel
                        / np.linalg.norm(self.robot_current_vel)
                        * self.robot_max_vel
                    )

                t = self.tf.getLatestCommonTime("/base_footprint", "/odom")
                position, quaternion = self.tf.lookupTransform(
                    "/base_footprint", "/odom", t
                )
                robot_offset_angle = tf.transformations.euler_from_quaternion(
                    quaternion
                )[2]

                print("offset_angle:", math.degrees(robot_offset_angle))

                angulo_velocidad = angle(
                    self.robot_current_vel, np.array([1, 0, 0], np.dtype("float64"))
                )

                print("angulo_velocidad:", math.degrees(angulo_velocidad))

                if self.robot_position[1] > self.current_waypoint[1]:
                    angulo_velocidad = angulo_velocidad - robot_offset_angle
                else:
                    angulo_velocidad = angulo_velocidad + robot_offset_angle

                print("angulo final (degree):", math.degrees(angulo_velocidad))

                if angulo_velocidad > 180:
                    angulo_velocidad = -180 + (angulo_velocidad - 180)

                vx = np.linalg.norm(self.robot_current_vel) * math.cos(angulo_velocidad)

                # w = np.linalg.norm(self.robot_current_vel) * math.sin(angulo_velocidad)

                w = self.pid_rotation(angulo_velocidad)

                cmd_vel_msg = Twist()
                cmd_vel_msg.linear.x = vx
                cmd_vel_msg.angular.z = w

                self.velocity_pub.publish(cmd_vel_msg)

                print("v lineal:", vx)
                print("w:", w)
                print("#####")


"""
obtencion de signo a partir de una numero
"""


def number_sign(n):
    if n == 0:
        return 0
    elif n > 0:
        return 1
    return -1


"""
producto punto de dos vectores
"""


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


"""
modulo de un vector
"""


def length(v):
    return math.sqrt(dotproduct(v, v))


"""
angulo entre dos vectores
"""


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


if __name__ == "__main__":
    rospy.init_node("sfm_drive_node")
    server = SocialForceModelDriveAction()
    rospy.spin()
