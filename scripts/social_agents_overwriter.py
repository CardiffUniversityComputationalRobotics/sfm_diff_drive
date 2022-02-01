#!/usr/bin/env python


import rospy
from pedsim_msgs.msg import AgentStates, AgentState
import tf
import copy


class AgentsOverwriter:
    def __init__(self):
        rospy.Subscriber(
            "/pedsim_simulator/simulated_agents", AgentStates, self.callback
        )
        # spin() simply keeps python from exiting until this node is stopped

        self.agents_pub = rospy.Publisher(
            "/pedsim_simulator/simulated_agents_overwritten", AgentStates, queue_size=1
        )

        self.agents_pos_to_add = [
            [-12.25, 7.5, 0, 100],
            [-12.25, 8.5, 0, 101],
            [-12.25, 9.5, 0, 102],
            [-12.25, 10.5, 0, 103],
        ]

    def main(self):
        while not rospy.is_shutdown():
            # actual_rmi_value = calculate_rmi(self.robot_odom, self.agents_list)
            # msg = Float32()
            # msg.data = actual_rmi_value
            # self.rmi_pub.publish(msg)
            # self.rate.sleep()
            pass

    def callback(self, data):
        agents_list = data.agent_states
        # new_agents_list = []
        # TODO: this should be done for all extra static agents to be added, a for is missing

        new_agent = AgentState()

        for agent in self.agents_pos_to_add:
            # print(agent[3])
            new_agent = AgentState()
            new_agent.id = agent[3]
            new_agent.header.stamp = rospy.Time.now()
            new_agent.header.frame_id = "world"
            new_agent.pose.position.x = agent[0]
            new_agent.pose.position.y = agent[1]

            quaternion = tf.transformations.quaternion_from_euler(0, 0, agent[2])

            new_agent.pose.orientation.x = quaternion[0]
            new_agent.pose.orientation.y = quaternion[1]
            new_agent.pose.orientation.z = quaternion[2]
            new_agent.pose.orientation.w = quaternion[3]
            # print("#######")
            # print(agents_list)
            agents_list.append(new_agent)
        # print(new_agents_list)

        data.agent_states = agents_list

        self.agents_pub.publish(data)


if __name__ == "__main__":
    rospy.init_node("agent_states_overwriter")

    agent_overwriter = AgentsOverwriter()
    agent_overwriter.main()
