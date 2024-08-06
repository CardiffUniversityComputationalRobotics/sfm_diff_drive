#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from rclpy.qos import QoSProfile
from rclpy.parameter import Parameter
from pedsim_msgs.msg import AgentStates, AgentGroups, LineObstacles
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from tf2_ros import TransformListener, Buffer
import numpy as np
import math
from sfm_diff_drive.action import SFMDrive
from move_base_msgs.action import MoveBase
from action_msgs.msg import GoalID
from std_msgs.msg import Bool

class SocialForceModelDriveAction(Node):

    def __init__(self):
        super().__init__('sfm_drive_node')

        # base variables
        self.goal_set = False

        self.xy_tolerance = 1

        self._action_name = "sfm_drive_node"

        self.move_base_client = self.create_client(MoveBase, 'move_base')

        self.agents_states_register = []
        self.agents_groups_register = []
        self.current_waypoint = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_position = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_orientation = np.array([0, 0, 0, 0], np.dtype("float64"))
        self.robot_current_vel = np.array([0, 0, 0], np.dtype("float64"))
        self.relaxation_time = 0.5
        self.laser_ranges = np.zeros(360)
        self.walls_range = []

        # nearest obstacle
        self.nearest_obstacle = np.array([0, 0, 0], np.dtype("float64"))

        self.agent_radius = 1
        self.force_sigma_obstacle = 0.8

        # for social force computing
        self.lambda_importance = 2
        self.gamma = 0.35
        self.n = 2
        self.n_prime = 3

        # constants for forces and other parameters
        self.declare_parameter('force_desired', 4.2)
        self.declare_parameter('force_social', 3.64)
        self.declare_parameter('force_obstacle', 35)
        self.declare_parameter('max_vel', 0.4)
        self.declare_parameter('max_vel_turn', 0.4)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('waypoints', [])
        self.declare_parameter('goal_path_topic', '')
        self.declare_parameter('social_agents_topic', '/pedsim_simulator/simulated_agents')
        self.declare_parameter('odom_topic', '/pepper/odom_groundtruth')
        self.declare_parameter('laser_topic', '/scan_filtered')
        self.declare_parameter('map_topic', '/projected_map')

        self.force_factor_desired = self.get_parameter('force_desired').value
        self.force_factor_social = self.get_parameter('force_social').value
        self.force_factor_obstacle = self.get_parameter('force_obstacle').value
        self.robot_max_vel = self.get_parameter('max_vel').value
        self.robot_max_turn_vel = self.get_parameter('max_vel_turn').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.waypoints = self.get_parameter('waypoints').value
        self.using_waypoints = False

        self.map = OccupancyGrid()

        # topic configs
        self.global_plan_topic = self.get_parameter('goal_path_topic').value
        self.agent_states_topic = self.get_parameter('social_agents_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.laser_topic = self.get_parameter('laser_topic').value
        self.map_topic = self.get_parameter('map_topic').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._as = ActionServer(
            self,
            SFMDrive,
            self._action_name,
            self.execute_cb,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Subscribers
        self.agents_states_subs = self.create_subscription(
            AgentStates,
            self.agent_states_topic,
            self.agents_state_callback,
            10
        )

        self.agents_groups_subs = self.create_subscription(
            AgentGroups,
            "/pedsim_simulator/simulated_groups",
            self.agents_groups_callback,
            10
        )

        self.robot_pos_subs = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.robot_pos_callback,
            10
        )

        self.laser_scan_subs = self.create_subscription(
            LaserScan,
            self.laser_topic,
            self.laser_scan_callback,
            10
        )

        self.obstacles_subs = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10
        )

        if self.global_plan_topic != '':
            self.global_plan_sub = self.create_subscription(
                Path,
                self.global_plan_topic,
                self.global_plan_callback,
                10
            )

        # Publishers
        self.velocity_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.goal_achieved_pub = self.create_publisher(Bool, "/goal_achieved", 10)

    def global_plan_callback(self, msg):
        self.waypoints = []
        for pose in msg.poses:
            self.waypoints.append([pose.pose.position.x, pose.pose.position.y])
        self.obstacle_map_processing()

    def check_goal_reached(self):
        if abs(np.linalg.norm(np.array([self.current_waypoint[0], self.current_waypoint[1]], np.dtype("float64")) - np.array([self.robot_position[0], self.robot_position[1]], np.dtype("float64")))) <= 0.3:
            if self.using_waypoints:
                if len(self.waypoints) != 1:
                    self.waypoints.pop(0)
                    self.current_waypoint = np.array([self.waypoints[0][0], self.waypoints[0][1], 0], np.dtype("float64"))
                else:
                    return True
            else:
                return True
        return False

    def execute_cb(self, goal_handle):
        self.get_logger().info("Starting social drive")
        r_sleep = self.create_rate(30)
        cancel_move_pub = self.create_publisher(GoalID, "/move_base/cancel", 1)
        cancel_msg = GoalID()
        cancel_move_pub.publish(cancel_msg)

        self.goal_set = True

        if len(self.waypoints) > 0:
            self.current_waypoint = np.array([self.waypoints[0][0], self.waypoints[0][1], 0], np.dtype("float64"))
            self.using_waypoints = True
        else:
            self.current_waypoint = np.array([goal_handle.request.goal.x, goal_handle.request.goal.y, goal_handle.request.goal.z], np.dtype("float64"))

        while not self.check_goal_reached():
            obstacle_complete_force = self.force_factor_obstacle * self.obstacle_force_walls()
            social_complete_force = self.force_factor_social * self.social_force()
            desired_complete_force = self.force_factor_desired * self.desired_force()
            complete_force = desired_complete_force + social_complete_force + obstacle_complete_force

            self.robot_current_vel = self.robot_current_vel + (complete_force / 25)
            speed = np.linalg.norm(self.robot_current_vel)

            if speed > self.robot_max_vel:
                self.robot_current_vel = self.robot_current_vel / np.linalg.norm(self.robot_current_vel) * self.robot_max_vel

            quaternion = (self.robot_orientation[0], self.robot_orientation[1], self.robot_orientation[2], self.robot_orientation[3])
            euler = self.tf_buffer.transformations.euler_from_quaternion(quaternion)

            robot_offset_angle = euler[2]

            if robot_offset_angle < 0:
                robot_offset_angle = 2 * math.pi + robot_offset_angle

            angulo_velocidad = math.atan2(self.robot_current_vel[0], self.robot_current_vel[1])

            if angulo_velocidad > 0 and angulo_velocidad < (math.pi / 2):
                angulo_velocidad = (math.pi / 2) - angulo_velocidad
            elif angulo_velocidad > (math.pi / 2):
                angulo_velocidad = (2 * math.pi) - angulo_velocidad + (math.pi / 2)
            elif angulo_velocidad < 0:
                angulo_velocidad = (math.pi / 2) - angulo_velocidad
            elif angulo_velocidad == 0:
                angulo_velocidad = math.pi / 2
            elif abs(angulo_velocidad) == (math.pi / 2):
                angulo_velocidad = math.pi * 3 / 2

            if robot_offset_angle > (angulo_velocidad + math.pi):
                yaw_error = angulo_velocidad + 2 * math.pi - robot_offset_angle
            elif angulo_velocidad > (robot_offset_angle + math.pi):
                yaw_error = robot_offset_angle + 2 * math.pi - angulo_velocidad
            else:
                yaw_error = angulo_velocidad - robot_offset_angle

            K_angular = 1.0
            vel = Twist()
            vel.linear.x = min(self.robot_max_vel, speed)
            vel.angular.z = K_angular * yaw_error
            self.velocity_pub.publish(vel)

            r_sleep.sleep()

        self.goal_set = False

        goal_handle.succeed()
        result = SFMDrive.Result()
        result.success = True
        self.goal_achieved_pub.publish(Bool(data=True))

        return result

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return rclpy.action.GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def agents_state_callback(self, msg):
        self.agents_states_register = msg.agent_states

    def agents_groups_callback(self, msg):
        self.agents_groups_register = msg.agent_groups

    def robot_pos_callback(self, msg):
        self.robot_position[0] = msg.pose.pose.position.x
        self.robot_position[1] = msg.pose.pose.position.y
        self.robot_position[2] = msg.pose.pose.position.z
        self.robot_orientation[0] = msg.pose.pose.orientation.x
        self.robot_orientation[1] = msg.pose.pose.orientation.y
        self.robot_orientation[2] = msg.pose.pose.orientation.z
        self.robot_orientation[3] = msg.pose.pose.orientation.w
        self.robot_current_vel[0] = msg.twist.twist.linear.x
        self.robot_current_vel[1] = msg.twist.twist.linear.y
        self.robot_current_vel[2] = msg.twist.twist.angular.z

    def laser_scan_callback(self, msg):
        self.laser_ranges = np.asarray(msg.ranges)

    def map_callback(self, msg):
        self.map = msg
        self.obstacle_map_processing()

    def obstacle_map_processing(self):
        self.walls_range = []
        array_2d = np.reshape(np.asarray(self.map.data), (self.map.info.height, self.map.info.width))

        for x in range(self.map.info.width):
            for y in range(self.map.info.height):
                if array_2d[y, x] == 100:
                    self.walls_range.append(Point(x=x * self.map.info.resolution, y=y * self.map.info.resolution))

    def desired_force(self):
        d = self.current_waypoint - self.robot_position
        d_mod = np.linalg.norm(d)
        f_desired = (d / d_mod) * self.robot_max_vel - self.robot_current_vel

        return f_desired / self.relaxation_time

    def social_force(self):
        force_social = np.zeros(3)
        for agent in self.agents_states_register:
            r_ab = self.robot_position - np.array([agent.pose.position.x, agent.pose.position.y, agent.pose.position.z])
            v_b = np.array([agent.twist.linear.x, agent.twist.linear.y, agent.twist.linear.z])
            v_ab = self.robot_current_vel - v_b

            r_ab_mod = np.linalg.norm(r_ab)
            r_ab_normalized = r_ab / r_ab_mod

            theta_ab = math.atan2(r_ab_normalized[1], r_ab_normalized[0])
            theta_vab = math.atan2(v_ab[1], v_ab[0])

            vector_angle = theta_ab - theta_vab
            angle_factor = self.lambda_importance + (1 - self.lambda_importance) * (1 + math.cos(vector_angle)) / 2

            exp_factor = math.exp(-r_ab_mod / self.gamma)
            force = angle_factor * exp_factor * (self.n + self.n_prime * (1 + math.cos(vector_angle))) * r_ab_normalized

            force_social = force_social + force

        return force_social

    def obstacle_force_walls(self):
        self.nearest_obstacle = [self.map.info.width, self.map.info.height]

        f_obstacle = np.zeros(3)

        for wall in self.walls_range:
            r_ao = np.array([wall.x, wall.y, 0], np.dtype("float64")) - self.robot_position
            distance = np.linalg.norm(r_ao)

            f = (self.robot_max_vel / distance) * math.exp(-distance / self.force_sigma_obstacle)

            if distance < 0.2:
                f_obstacle = -self.robot_position + self.current_waypoint

            f_obstacle = f_obstacle + f

        return f_obstacle

def main(args=None):
    rclpy.init(args=args)
    sfm_drive_action = SocialForceModelDriveAction()

    try:
        rclpy.spin(sfm_drive_action)
    except KeyboardInterrupt:
        pass

    sfm_drive_action.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
