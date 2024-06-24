import rclpy
from rclpy.node import Node
from rclpy.task import Future

import sys
import math
from math import pow, sin, cos, atan2, sqrt
import random
from  functools import partial

from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import SetPen, Spawn, Kill


class Move2GoalNode(Node):
    def __init__(self, goal_pose, tolerance=1, spawn_tolerance=0.05, max_turtles_spawned=3, K1=1, K2=3):
        super().__init__('move2goal')

        self.goal_pose = goal_pose
        self.tolerance = tolerance
        self.current_pose = None

        self.pub_dict = {}
        self.sub_dict = {}

        self.spawn_poses = {}
        self.spawn_tolerance = spawn_tolerance
        self.spawn_goal_poses = {}
        self.spawn_pens = {}

        self.STATES = ['WRITING', 'ANGRY', 'RETURNING']
        self.K2 = K2
        self.K1 = K1
        self.max_turtles_spawned = max_turtles_spawned 

        self.state = self.STATES[0]
        self.old_position = None
        self.turtle_following = None
        self.last_pen = 0
        
        self.spawn_client = self.create_client(Spawn, '/spawn')
        self.pen_client = self.create_client(SetPen, '/turtle1/set_pen')
        self.kill_client = self.create_client(Kill, '/kill')

        self.vel_publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        self.pose_subscriber = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        
        self.timer_spawn = self.create_timer(3, self.spawn_turtle)
        self.timer_spawns_move = self.create_timer(0.1, self.move_spawns_callback)


    def wait_for_pen_service(self):
        while not self.pen_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('pen service not available, waiting again...')
        self.get_logger().info('service available')
        self.turn_pen(0)

    def turn_spawn_pen_off(self,spawn):
        self.spawn_pens[spawn].call_async(SetPen.Request(r=0,g=0,b=0,width=0,off=1))

    def gen_spawn(self,i):
        func = partial(self.spawns_pose_callback, spawn=i)
        self.sub_dict[i] = self.create_subscription(Pose, '/spawn'+str(i)+'/pose', func, 10)
        self.pub_dict[i] = self.create_publisher(Twist, '/spawn'+str(i)+'/cmd_vel', 10)
        self.spawn_poses[i] = None
        self.spawn_goal_poses[i] = None
        self.spawn_pens[i] = self.create_client(SetPen, '/spawn'+str(i)+'/set_pen')

    def predict_pose(self, pose, seconds):
        if pose.angular_velocity != 0:
            r = pose.linear_velocity / pose.angular_velocity
            new_theta = pose.theta + pose.angular_velocity * seconds
            new_x = pose.x + r * (sin(new_theta) -  sin(pose.theta))
            new_y = pose.y - r * (cos(new_theta) - cos(pose.theta))
        else:
            new_x = pose.x + pose.linear_velocity * seconds * cos(pose.theta)
            new_y = pose.y + pose.linear_velocity * seconds * sin(pose.theta)
        return new_x, new_y
  

    def move_spawns_callback(self):
        for i,pose in self.spawn_poses.items():
            if pose is None or i not in self.pub_dict:
                continue
            if self.spawn_goal_poses[i] is None:
                self.spawn_goal_poses[i] = pose
            if self.euclidean_distance(self.spawn_goal_poses[i], pose) < self.spawn_tolerance:
                self.spawn_goal_poses[i].x = float(min(max(1,self.spawn_goal_poses[i].x + random.randint(-1,1)),10))
                self.spawn_goal_poses[i].y = float(min(max(1,self.spawn_goal_poses[i].y + random.randint(-1,1)),10))
            self.turn_spawn_pen_off(i)
            cmd_vel = Twist()
            cmd_vel.linear.x = self.linear_vel(self.spawn_goal_poses[i], pose, 1)
            cmd_vel.angular.z = self.angular_vel(self.spawn_goal_poses[i], pose, 6)
            self.pub_dict[i].publish(cmd_vel)


    def spawns_pose_callback(self, msg, spawn):
        self.spawn_poses[spawn] = msg
        self.spawn_poses[spawn].x = round(self.spawn_poses[spawn].x, 4)
        self.spawn_poses[spawn].y = round(self.spawn_poses[spawn].y, 4)

    def get_spawned_indexes(self):
        temp_set = set()
        for k,v in self.pub_dict.items():
            if v is not None:
                temp_set.add(k)
        return temp_set

    def get_spawn_new_index(self):
        temp_set = self.get_spawned_indexes()
        if len(temp_set) < self.max_turtles_spawned:
            for i in range(self.max_turtles_spawned):
                if i not in temp_set:
                    return i
        return None

    def spawn_turtle(self):
        index = self.get_spawn_new_index()
        if not index is None:
            req = Spawn.Request()
            req.x = random.uniform(1, 10)
            req.y = random.uniform(1, 10)
            req.theta = random.uniform(0, 2*math.pi)
            req.name = 'spawn'+str(index)
            self.spawn_client.call_async(req)
            self.gen_spawn(index)


    def set_pen(self, r, g, b, width, off):
        return self.pen_client.call_async(SetPen.Request(r=r, g=g, b=b, width=width, off=off))

    def turn_pen(self,ON, save_this=True):
        if save_this:
            self.last_pen = ON
        if ON == 1:
            return self.set_pen(255, 255, 255, 5, 0)
        else:
            return self.set_pen(0, 0, 0, 0, 1)
    
    def start_moving(self):
        self.timer = self.create_timer(0.1, self.move_callback)
        self.done_future = Future()
        return self.done_future
        
    def pose_callback(self, msg):
        self.current_pose = msg
        self.current_pose.x = round(self.current_pose.x, 4)
        self.current_pose.y = round(self.current_pose.y, 4)

    def get_closest_turtle(self):
        min_dist = None
        min_turtle = None
        for i,pose in self.spawn_poses.items():
            if pose is None:
                continue
            dist = self.euclidean_distance(pose, self.current_pose)
            if dist < self.K2:
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_turtle = i
        return min_turtle
    
    def follow_turtle(self):
        if self.euclidean_distance(self.spawn_poses[self.turtle_following], self.current_pose) < self.K1:

            self.kill_client.call_async(Kill.Request(name='spawn'+str(self.turtle_following)))
            self.destroy_subscription(self.spawn_pens[self.turtle_following])
            self.destroy_subscription(self.pub_dict[self.turtle_following])
            self.destroy_subscription(self.sub_dict[self.turtle_following])

            self.spawn_poses[self.turtle_following] = None
            self.spawn_goal_poses[self.turtle_following] = None
            self.spawn_pens[self.turtle_following] = None
            self.pub_dict[self.turtle_following] = None
            self.sub_dict[self.turtle_following] = None
            self.turtle_following = None
            self.state = self.STATES[2]
            self.move_to_goal()
        else:
            pred_pose = Pose()
            pred_pose.x, pred_pose.y = self.predict_pose(self.spawn_poses[self.turtle_following], 0.5)
            pred_pose = self.truncate_path(self.current_pose, pred_pose, self.K1)
            cmd_vel = Twist() 
            cmd_vel.linear.x = self.linear_vel(pred_pose, self.current_pose)
            cmd_vel.angular.z = self.angular_vel(pred_pose, self.current_pose)
            self.vel_publisher.publish(cmd_vel)


    def move_callback(self):
        if self.current_pose is None:
            return
        if self.state == self.STATES[0]:
            closest_turtle = self.get_closest_turtle()
            if closest_turtle is not None: 
                self.state = self.STATES[1]
                self.old_position = self.current_pose
                self.turtle_following = closest_turtle
                self.turn_pen(0,False)
                self.follow_turtle()
            else:
                self.move_to_goal()
        elif self.state == self.STATES[1]:
            self.follow_turtle()
        else: 
            closest_turtle = self.get_closest_turtle()
            if closest_turtle is not None: 
                self.state = self.STATES[1]
                self.turtle_following = closest_turtle
                self.follow_turtle()
            else:
                if self.euclidean_distance(self.old_position, self.current_pose) >= self.tolerance:
                    cmd_vel = Twist()
                    goto_pos = self.truncate_path(self.current_pose, self.old_position) 
                    cmd_vel.linear.x = self.linear_vel(goto_pos, self.current_pose)
                    cmd_vel.angular.z = self.angular_vel(goto_pos, self.current_pose)
                    self.vel_publisher.publish(cmd_vel)
                else:
                    self.state = self.STATES[0]
                    self.turn_pen(self.last_pen)
                    self.move_to_goal()


    def truncate_path(self, from_pose, to_pose, walk_measure=1):
        if self.euclidean_distance(from_pose, to_pose) < walk_measure:
            return to_pose
        else:
            new_pose = Pose()
            new_pose.x = from_pose.x + walk_measure * (to_pose.x - from_pose.x) / self.euclidean_distance(from_pose, to_pose)
            new_pose.y = from_pose.y + walk_measure * (to_pose.y - from_pose.y) / self.euclidean_distance(from_pose, to_pose)
            return new_pose

    def move_to_goal(self):
        if self.euclidean_distance(self.goal_pose, self.current_pose) >= self.tolerance:
            goto_pos = self.truncate_path(self.current_pose, self.goal_pose)
            cmd_vel = Twist() 
            cmd_vel.linear.x = self.linear_vel(goto_pos, self.current_pose)
            cmd_vel.angular.z = self.angular_vel(goto_pos, self.current_pose)
            self.vel_publisher.publish(cmd_vel)
        else:
            cmd_vel = Twist() 
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.vel_publisher.publish(cmd_vel)
            self.done_future.set_result(True)        


    def euclidean_distance(self, goal_pose, current_pose):
        """Euclidean distance between current pose and the goal."""
        return sqrt(pow((goal_pose.x - current_pose.x), 2) +
                    pow((goal_pose.y - current_pose.y), 2))

    def angular_difference(self, goal_theta, current_theta):
        """Compute shortest rotation from orientation current_theta to orientation goal_theta"""
        return atan2(sin(goal_theta - current_theta), cos(goal_theta - current_theta))

    def linear_vel(self, goal_pose, current_pose, constant=1.5):
        """See video: https://www.youtube.com/watch?v=Qh15Nol5htM."""
        return constant * self.euclidean_distance(goal_pose, current_pose)

    def steering_angle(self, goal_pose, current_pose):
        """See video: https://www.youtube.com/watch?v=Qh15Nol5htM."""
        return atan2(goal_pose.y - current_pose.y, goal_pose.x - current_pose.x)

    def angular_vel(self, goal_pose, current_pose, constant=6):
        """See video: https://www.youtube.com/watch?v=Qh15Nol5htM."""
        goal_theta = self.steering_angle(goal_pose, current_pose)
        return constant * self.angular_difference(goal_theta, current_pose.theta)


def initialize_coords():
    U,S,I = [],[(7,10),(5,10),(5,5),(7,5),(7,1),(5,1)],[]
    for i in range(10,0,-3):
        U.append((1,i))
    for i in range(1,11,3):
        U.append((3,i))
    for i in range(1,11,3):
        I.append((9,i))
    return [U,S,I]


def main():

    rclpy.init(args=sys.argv)
    goal_pose = Pose()
    node = Move2GoalNode(Pose(), tolerance=0.05, spawn_tolerance=1, max_turtles_spawned=3, K1=1, K2=3)
    node.wait_for_pen_service()

    for L in initialize_coords():
        for (x,y) in L:
            goal_pose.x = float(x)
            goal_pose.y = float(y)
            node.goal_pose = goal_pose
            done = node.start_moving()
            rclpy.spin_until_future_complete(node, done)
            node.turn_pen(1).done()
        node.turn_pen(0).done()


if __name__ == '__main__':
    main()

