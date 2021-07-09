#!/usr/bin/env python

from numpy.core.defchararray import multiply
import rospy

from pedsim_msgs.msg import AgentStates, AgentGroups
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from tf import TransformListener
import tf
import numpy as np
import math

np.version.version


class SocialForceModelDrive:
    def __init__(self):
        rospy.init_node("sfm_force_computer_publisher")

        # base variables
        self.goal_set = False

        self.agents_states_register = []
        self.agents_groups_register = []
        self.current_waypoint = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_position = np.array([0, 0, 0], np.dtype("float64"))

        self.robot_max_vel = 0.5
        self.robot_current_vel = np.array([0, 0, 0], np.dtype("float64"))
        self.relaxation_time = 0.5
        self.laser_ranges = np.zeros(360)

        self.rate = rospy.Rate(25)
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
        self.force_factor_obstacle = 5

        self.tf = TransformListener()

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

        self.current_waypoint_subs = rospy.Subscriber(
            "/pedsim_robot/waypoint", Point, self.waypoint_callback
        )

        self.laser_scan_subs = rospy.Subscriber(
            "/scan_filtered", LaserScan, self.laser_scan_callback
        )

        #! publishers
        self.velocity_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    # * callbacks
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

    """
    callback para obtener posicion del waypoint
    """

    def waypoint_callback(self, data):
        self.goal_set = True
        self.current_waypoint = np.array([data.x, data.y, data.z], np.dtype("float64"))

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

        # evaluo si el indice menor no existe
        if math.isnan(diff_robot_laser.min()):
            return np.array([0, 0, 0], np.dtype("float64"))
        else:
            min_index = np.where(diff_robot_laser == diff_robot_laser.min())[0][0]

            if diff_robot_laser[min_index] < 0.3:

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
                print("laser_pos:", laser_pos)

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

                angulo_velocidad = (
                    angle(
                        np.array([1, 0, 0], np.dtype("float64")),
                        self.robot_current_vel,
                    )
                    - robot_offset_angle
                )

                print("angulo velocidad:", math.degrees(angulo_velocidad))

                vx = np.linalg.norm(self.robot_current_vel) * math.cos(angulo_velocidad)

                w = np.linalg.norm(self.robot_current_vel) * math.sin(angulo_velocidad)

                cmd_vel_msg = Twist()
                cmd_vel_msg.linear.x = vx
                cmd_vel_msg.angular.z = -w

                self.velocity_pub.publish(cmd_vel_msg)

                print("v lineal:", vx)
                print("w:", -w)
                print("#####")

            self.rate.sleep()


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
    forces_computer = SocialForceModelDrive()
    forces_computer.run()
