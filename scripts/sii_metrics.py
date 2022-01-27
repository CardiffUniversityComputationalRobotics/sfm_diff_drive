#!/usr/bin/env python

import rospy
from pedsim_msgs.msg import AgentStates
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import numpy as np
import math


class SIIMetricsPublisher(object):
    def __init__(self):

        # subscribers

        self.agents_states_subs = rospy.Subscriber(
            "/pedsim_simulator/simulated_agents_overwritten",
            AgentStates,
            self.agents_state_callback,
        )

        self.robot_pos_subs = rospy.Subscriber(
            "/pepper/odom_groundtruth",
            Odometry,
            self.robot_pos_callback,
        )

        # publishers
        self.sii_metric_pub = rospy.Publisher("/sii_value", Float64, queue_size=10)

        self.sii_value_msg = Float64()
        self.sii_value_msg.data = 0

        """d_c: is the desirable value of the distance between the robot and the 
        agents, can be around 0.45m and 1.2m according to Hall depending on the
        culture
        """
        self.d_c = 1.2
        self.sigma_p = self.d_c / 2
        self.final_sigma = math.sqrt(2) * self.sigma_p

        self.agents_states_register = []
        self.robot_position = np.array([0, 0, 0], np.dtype("float64"))
        self.rate = rospy.Rate(30)

    def agents_state_callback(self, data):
        self.agents_states_register = data.agent_states

    def robot_pos_callback(self, data):
        data_position = data.pose.pose.position
        self.robot_position = np.array(
            [data_position.x, data_position.y, data_position.z], np.dtype("float64")
        )

    def run(self):
        while not rospy.is_shutdown():
            self.sii_value_msg.data = 0
            sii_value = 0

            for agent in self.agents_states_register:
                sii_value = math.pow(
                    math.e,
                    -(
                        math.pow(
                            (self.robot_position[0] - agent.pose.position.x)
                            / (self.final_sigma),
                            2,
                        )
                        + math.pow(
                            (self.robot_position[1] - agent.pose.position.y)
                            / (self.final_sigma),
                            2,
                        )
                    ),
                )
                # print(self.robot_position[0] - agent.pose.position.x)
                # print(self.robot_position[1] - agent.pose.position.y)
                # print("agentid: ", agent.id)
                # print("sii value:", sii_value)
                if sii_value > self.sii_value_msg.data:
                    # print("last value: ", self.sii_value_msg.data)
                    self.sii_value_msg.data = sii_value
            # print("max value:", self.sii_value_msg.data)
            # print("#############")
            self.sii_metric_pub.publish(self.sii_value_msg)
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("sii_metrics_node")
    server = SIIMetricsPublisher()
    server.run()
