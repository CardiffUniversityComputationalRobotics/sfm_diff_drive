#!/usr/bin/env python3
import time

from numpy.linalg.linalg import norm
import rospy

from pedsim_msgs.msg import AgentStates, AgentGroups
from move_base_msgs.msg import MoveBaseActionGoal
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import numpy as np


class SocialForceModelDrive:
    def __init__(self):
        rospy.init_node("sfm_force_computer_publisher")

        # base variables
        self.agents_states_register = None
        self.agents_groups_register = None
        self.current_waypoint = np.array([0, 0, 0])
        self.robot_position = np.array([0, 0, 0])

        self.robot_max_vel = 0.3
        self.robot_current_vel = 0
        self.relaxation_time = 0.5
        self.laser_ranges = np.zeros(360)

        self.rate = rospy.Rate(100)

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
            "/move_base/goal", MoveBaseActionGoal, self.waypoint_callback
        )

        self.laser_scan_subs = rospy.Subscriber(
            "/scan_filtered", LaserScan, self.laser_scan_callback
        )

        #! publishers

    # * callbacks
    def laser_scan_callback(self, data):
        self.laser_ranges = data.ranges

    def robot_pos_callback(self, data):
        data = data.pose.pose.position
        self.robot_position = np.array([data.x, data.y, data.z])

    def agents_state_callback(self, data):
        self.agents_states_register = data

    def agents_groups_callback(self, data):
        self.agents_groups_register = data

    def waypoint_callback(self, data):
        data = data.goal.target_pose.pose.position
        self.current_waypoint = np.array([data.x, data.y, data.z])

    # * force functions
    def desired_force(self):
        print("Waypoint actual")
        print(self.current_waypoint)
        print("posicion actual de robot")
        print(self.robot_position)
        desired_direction = self.current_waypoint - self.robot_position
        desired_direction_vec_norm = np.linalg.norm(desired_direction)
        if desired_direction_vec_norm != 0:
            norm_desired_direction = desired_direction / np.linalg.norm(
                desired_direction
            )
        else:
            norm_desired_direction = np.array([0, 0, 0])
        desired_force = (
            norm_desired_direction * self.robot_max_vel - self.robot_current_vel
        ) / self.relaxation_time

        return desired_force

    def obstacle_force(self):
        # TODO: finish this function
        pass

    def run(self):
        while not rospy.is_shutdown():
            self.desired_force()
            self.rate.sleep()


if __name__ == "__main__":
    forces_computer = SocialForceModelDrive()
    forces_computer.run()
