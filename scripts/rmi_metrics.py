from typing import List
import rospy
from pedsim_msgs.msg import AgentStates, AgentState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np
import math
import tf


class RMIMetricsPublisher:
    def __init__(self):

        self.agents_list = []
        self.robot_odom = None

        #! subscribers
        self.agents_states_subs = rospy.Subscriber(
            "/pedsim_simulator/simulated_agents",
            AgentStates,
            self.agents_state_callback,
        )

        self.robot_pos_subs = rospy.Subscriber(
            "/pedsim_simulator/robot_position",
            Odometry,
            self.robot_pos_callback,
        )

        #! publishers

        self.rmi_pub = rospy.Publisher("/rmi", Float32, queue_size=10)

        self.rate = rospy.Rate(10)

    """
    callback para agarrar datos de posicion del robot
    """

    def robot_pos_callback(self, data):
        self.robot_odom = data

    """
    callback para obtener lista de info de agentes
    """

    def agents_state_callback(self, data):
        self.agents_list = data.agent_states

    def main(self):
        while not rospy.on_shutdown:
            actual_rmi_value = calculate_rmi(self.robot_odom, self.agents_list)
            msg = Float32()
            msg.data = actual_rmi_value
            self.rmi_pub.publish(msg)
            self.rate.sleep()


def calculate_rmi(robot_odometry: Odometry, agents_list: List[AgentState]):
    last_rmi = 0

    rmi_value = lambda v_r, beta, v_a, alpha, x_agent, y_agent, x_robot, y_robot: (
        2 + v_r * np.cos(beta) + v_a * np.cos(alpha)
    ) / (np.sqrt(np.pow(x_agent - x_robot, 2) + np.pow(y_agent - y_robot, 2)))

    v_r = np.sqrt(
        np.pow(robot_odometry.twist.twist.linear.x, 2)
        + np.pow(robot_odometry.twist.twist.linear.y, 2)
    )

    for agent in agents_list:

        beta = math.atan2(
            agent.pose.position.y - robot_odometry.pose.pose.position.y,
            agent.pose.position.x - robot_odometry.pose.pose.position.x,
        )

        # beta = np.arctan2((state_r2->values[1] - agentState.pose.position.y),
        #                            (state_r2->values[0] - agentState.pose.position.x)))

        quaternion = (
            robot_odometry.pose.orientation.x,
            robot_odometry.pose.orientation.y,
            robot_odometry.pose.orientation.z,
            robot_odometry.pose.orientation.w,
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]

        robot_angle = math.pi / 2 + yaw

        beta = abs(robot_angle - beta)

        # if robot_angle > beta:
        #     robot_angle - beta
        # elif robot_angle < beta:
        #     beta - robot_angle
        # else:
        #     beta = 0

        v_a = np.sqrt(np.pow(agent.twist.linear.x, 2) + np.pow(agent.twist.linear.y, 2))

        alpha = math.atan2(
            robot_odometry.pose.pose.position.y - agent.pose.position.y,
            robot_odometry.pose.pose.position.x - agent.pose.position.x,
        )

        # beta = np.arctan2((state_r2->values[1] - agentState.pose.position.y),
        #                            (state_r2->values[0] - agentState.pose.position.x)))

        quaternion = (
            agent.pose.orientation.x,
            agent.pose.orientation.y,
            agent.pose.orientation.z,
            agent.pose.orientation.w,
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]

        agent_angle = math.pi / 2 + yaw

        alpha = abs(agent_angle - alpha)

        current_rmi = rmi_value(
            v_r,
            beta,
            v_a,
            alpha,
            agent.pose.position.x,
            agent.pose.position.y,
            robot_odometry.pose.pose.position.x,
            robot_odometry.pose.pose.position.y,
        )

        if current_rmi > last_rmi:
            last_rmi = current_rmi

    return last_rmi


if __name__ == "__main__":
    rospy.init_node("sii_metrics_node")
    rmi_publisher_node = RMIMetricsPublisher()
    rmi_publisher_node.main()
