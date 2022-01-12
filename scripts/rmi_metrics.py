import rospy
from pedsim_msgs.msg import AgentStates, AgentState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np


class RMIMeasure:
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
            self.rmi_pub.publish(actual_rmi_value)
            self.rate.sleep()


def calculate_rmi(robot_odometry, agents_list):
    last_rmi = 0

    rmi_value = lambda v_r, beta, v_a, alpha, x_agent, y_agent, x_robot, y_robot: (
        2 + v_r * np.cos(beta) + v_a * np.cos(alpha)
    ) / (np.sqrt(np.pow(x_agent - x_robot, 2) + np.pow(y_agent - y_robot, 2)))

    v_r = np.sqrt(
        np.pow(robot_odometry.twist.twist.linear.x, 2)
        + np.pow(robot_odometry.twist.twist.linear.y, 2)
    )

    for agent in agents_list:

        beta = 1  # angulo de la orientacion del robot a la posicion del agente

        v_a = np.sqrt(np.pow(agent.twist.linear.x, 2) + np.pow(agent.twist.linear.y, 2))

        alpha = 1

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
