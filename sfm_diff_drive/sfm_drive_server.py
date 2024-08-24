import rclpy
from rclpy.node import Node
from pedsim_msgs.msg import AgentStates, AgentGroups
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
import numpy as np
import math
from std_msgs.msg import Bool
from tf_transformations import euler_from_quaternion
from esc_move_base_msgs.msg import Path2D


class SocialForceModelDriveAction(Node):

    def __init__(self):
        super().__init__("sfm_drive_node")

        # base variables

        self.agents_states_register = []
        self.agents_groups_register = []
        self.current_waypoint = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_position = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_orientation = np.array([0, 0, 0, 0], np.dtype("float64"))
        self.robot_current_vel = np.array([0, 0, 0], np.dtype("float64"))
        self.relaxation_time = 0.5
        self.laser_ranges = np.zeros(360)

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
        self.declare_parameter("force_desired", 4.2)
        self.declare_parameter("force_social", 3.64)
        self.declare_parameter("force_obstacle", 35.0)
        self.declare_parameter("max_vel", 0.4)
        self.declare_parameter("max_vel_turn", 0.4)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("waypoints", [])
        self.declare_parameter("goal_path_topic", "")
        self.declare_parameter(
            "social_agents_topic", "/pedsim_simulator/simulated_agents"
        )
        self.declare_parameter("odom_topic", "/pepper/odom_groundtruth")
        self.declare_parameter("laser_topic", "/scan_filtered")
        self.declare_parameter("map_topic", "/projected_map")

        self.force_factor_desired = self.get_parameter("force_desired").value
        self.force_factor_social = self.get_parameter("force_social").value
        self.force_factor_obstacle = self.get_parameter("force_obstacle").value
        self.robot_max_vel = self.get_parameter("max_vel").value
        self.robot_max_turn_vel = self.get_parameter("max_vel_turn").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.waypoints = self.get_parameter("waypoints").value
        self.using_waypoints = False

        self.map = OccupancyGrid()

        # topic configs
        self.global_plan_topic = self.get_parameter("goal_path_topic").value
        self.agent_states_topic = self.get_parameter("social_agents_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.laser_topic = self.get_parameter("laser_topic").value
        self.map_topic = self.get_parameter("map_topic").value

        self.goal = None

        # Subscribers
        self.agents_states_subs = self.create_subscription(
            AgentStates, self.agent_states_topic, self.agents_state_callback, 10
        )

        self.agents_groups_subs = self.create_subscription(
            AgentGroups,
            "/pedsim_simulator/simulated_groups",
            self.agents_groups_callback,
            10,
        )

        self.robot_pos_subs = self.create_subscription(
            Odometry, self.odom_topic, self.robot_pos_callback, 10
        )

        self.laser_scan_subs = self.create_subscription(
            LaserScan, self.laser_topic, self.laser_scan_callback, 10
        )

        self.obstacles_subs = self.create_subscription(
            OccupancyGrid, self.map_topic, self.map_callback, 10
        )

        self.goal_subs = self.create_subscription(
            PoseStamped, "/goal_pose", self.global_goal_callback, 5
        )

        if self.global_plan_topic != "":
            self.global_plan_sub = self.create_subscription(
                Path2D, self.global_plan_topic, self.global_plan_callback, 10
            )

        # Publishers
        self.velocity_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.goal_achieved_pub = self.create_publisher(Bool, "/goal_reached", 10)

    def global_goal_callback(self, msg):
        self.goal = msg

    def global_plan_callback(self, msg: Path2D):
        self.waypoints = []

        if len(msg.waypoints) > 1:

            got_waypoints = msg.waypoints

            got_waypoints.reverse()

            for pos in got_waypoints:
                if (
                    math.sqrt(
                        (pos.x - self.robot_position[0]) ** 2
                        + (pos.y - self.robot_position[1]) ** 2
                    )
                    > 0.4
                ):
                    self.waypoints.append([pos.x, pos.y])
                else:
                    break

            self.waypoints.reverse()

        elif len(msg.waypoints) == 1:
            self.waypoints.append([msg.waypoints[-1].x, msg.waypoints[-1].y])
        # print(self.waypoints)
        self.obstacle_map_processing()

    def agents_state_callback(self, msg):
        self.agents_states_register = msg.agent_states

    def agents_groups_callback(self, msg):
        # self.agents_groups_register = msg.agent_groups
        pass

    def robot_pos_callback(self, msg: Odometry):
        """
        callback para agarrar datos de posicion del robot
        """
        data_position = msg.pose.pose.position
        self.robot_position = np.array(
            [data_position.x, data_position.y, data_position.z], np.dtype("float64")
        )

        self.robot_orientation = np.array(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ],
            np.dtype("float64"),
        )

    def laser_scan_callback(self, msg):
        self.laser_ranges = np.asarray(msg.ranges)

    def map_callback(self, msg):
        self.map = msg

    # define MAP_INDEX(map, i, j) ((i) + (j) * map.size_x)
    def map_index(self, size_x, i, j):
        return i + j * size_x

    def map_wx(self, origin_x, size_x, scale, i):
        return origin_x + (i - size_x / 2) * scale

    def map_wy(self, origin_y, size_y, scale, j):
        return origin_y + (j - size_y / 2) * scale

    def obstacle_map_processing(self):
        cur_nearest_obs = (0, 0)
        cur_nearest_dist = 1000000000

        map_size_x = self.map.info.width
        map_size_y = self.map.info.height
        map_scale = self.map.info.resolution
        map_origin_x = self.map.info.origin.position.x + (map_size_x / 2) * map_scale
        map_origin_y = self.map.info.origin.position.y + (map_size_y / 2) * map_scale

        # map_origin_x = 0 + (map_size_x / 2) * map_scale
        # map_origin_y = 0 + (map_size_y / 2) * map_scale

        for j in range(0, map_size_y, 2):
            for i in range(0, map_size_x, 2):
                if self.map.data[self.map_index(map_size_x, i, j)] == 100:
                    w_x = self.map_wx(map_origin_x, map_size_x, map_scale, i)
                    w_y = self.map_wy(map_origin_y, map_size_y, map_scale, j)
                    cur_dist = np.power(w_x - self.robot_position[0], 2) + np.power(
                        w_y - self.robot_position[1], 2
                    )

                    if cur_dist < cur_nearest_dist:
                        cur_nearest_dist = cur_dist
                        cur_nearest_obs = (w_x, w_y)
                        # print(cur_dist)

        self.nearest_obstacle[0] = cur_nearest_obs[0]
        self.nearest_obstacle[1] = cur_nearest_obs[1]

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
        return desired_force

    def social_force(self):
        """
        funcion para obtener la fuerzas sociales de los alrededores
        """

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

            # theta = angle(interaction_direction, diff_direction)

            theta = math.atan2(diff_direction[1], diff_direction[0]) - math.atan2(
                interaction_direction[1], interaction_direction[0]
            )

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

    def obstacle_force_walls(self):
        """
        funcion para obtener la fuerza de el obstaculo mas cercano conociendo la posicion exacta de todos ellos de manera estatica
        """

        diff_robot_obstacle = np.sqrt(
            np.power(self.nearest_obstacle[0] - self.robot_position[0], 2)
            + np.power(self.nearest_obstacle[1] - self.robot_position[1], 2)
        )

        nearest_obstacle_temp = self.robot_position - self.nearest_obstacle

        obstacle_vec_norm = np.linalg.norm(nearest_obstacle_temp)
        if obstacle_vec_norm != 0:
            norm_obstacle_direction = nearest_obstacle_temp / obstacle_vec_norm
        else:
            norm_obstacle_direction = np.array([0, 0, 0], np.dtype("float64"))

        distance = diff_robot_obstacle - self.agent_radius
        force_amount = math.exp(-distance / self.force_sigma_obstacle)
        final_rep_force = force_amount * norm_obstacle_direction
        # print(final_rep_force)
        return final_rep_force

    def movement_callback(self):
        self.get_logger().info("Starting social force model drive")

        while rclpy.ok():
            rclpy.spin_once(self)
            if len(self.waypoints) > 0:
                if (
                    math.sqrt(
                        math.pow(self.robot_position[0] - self.goal.pose.position.x, 2)
                    )
                    + math.sqrt(
                        math.pow(self.robot_position[1] - self.goal.pose.position.y, 2)
                    )
                    < 0.5
                ):
                    reached_goal = Bool()
                    reached_goal.data = True
                    self.goal_achieved_pub.publish(reached_goal)
                    vel = Twist()
                    vel.linear.x = 0.0
                    vel.angular.z = 0.0
                    self.velocity_pub.publish(vel)

                else:

                    while not len(self.waypoints) <= 1:
                        if (
                            math.sqrt(
                                math.pow(
                                    self.robot_position[0] - self.waypoints[0][0], 2
                                )
                            )
                            + math.sqrt(
                                math.pow(
                                    self.robot_position[1] - self.waypoints[0][1], 2
                                )
                            )
                            > 0.2
                        ):
                            break
                        else:
                            self.waypoints.pop(0)
                    self.current_waypoint = np.array(
                        [self.waypoints[0][0], self.waypoints[0][1], 0],
                        np.dtype("float64"),
                    )

                    obstacle_complete_force = (
                        self.force_factor_obstacle * self.obstacle_force_walls()
                    )
                    social_complete_force = (
                        self.force_factor_social * self.social_force()
                    )
                    desired_complete_force = (
                        self.force_factor_desired * self.desired_force()
                    )
                    complete_force = (
                        desired_complete_force
                        + social_complete_force
                        + obstacle_complete_force
                    )

                    self.robot_current_vel = self.robot_current_vel + (
                        complete_force / 25
                    )
                    speed = np.linalg.norm(self.robot_current_vel)

                    if speed > self.robot_max_vel:
                        self.robot_current_vel = (
                            self.robot_current_vel
                            / np.linalg.norm(self.robot_current_vel)
                            * self.robot_max_vel
                        )

                    quaternion = (
                        self.robot_orientation[0],
                        self.robot_orientation[1],
                        self.robot_orientation[2],
                        self.robot_orientation[3],
                    )
                    euler = euler_from_quaternion(quaternion)

                    robot_offset_angle = euler[2]

                    if robot_offset_angle < 0:
                        robot_offset_angle = 2 * math.pi + robot_offset_angle

                    angulo_velocidad = math.atan2(
                        self.robot_current_vel[0], self.robot_current_vel[1]
                    )

                    if angulo_velocidad > 0 and angulo_velocidad < (math.pi / 2):
                        angulo_velocidad = (math.pi / 2) - angulo_velocidad
                    elif angulo_velocidad > (math.pi / 2):
                        angulo_velocidad = (
                            (2 * math.pi) - angulo_velocidad + (math.pi / 2)
                        )
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
                # time.sleep(1)


def main(args=None):
    rclpy.init(args=args)
    sfm_drive_server = SocialForceModelDriveAction()

    try:
        sfm_drive_server.movement_callback()
        rclpy.spin(sfm_drive_server)
    except KeyboardInterrupt:
        pass

    sfm_drive_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


def number_sign(n):
    if n == 0:
        return 0
    elif n > 0:
        return 1
    return -1
