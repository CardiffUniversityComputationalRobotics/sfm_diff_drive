import rospy
from pedsim_msgs.msg import AgentStates, AgentState
from nav_msgs.msg import Odometry

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


def calculate_rmi():
    pass



