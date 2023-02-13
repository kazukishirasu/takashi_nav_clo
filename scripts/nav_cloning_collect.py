#!/usr/bin/env python3
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from geometry_msgs.msg import PoseWithCovarianceStamped,Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_net import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseStamped

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState

import math
import tf

from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import sys

class cource_following_learning_node:
    def __init__(self):
        rospy.init_node('cource_following_learning_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.min_distance = 0.0
        self.action = 0.0
        self.vel = Twist()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.init = True
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.name = '02'
        self.goal_offset = 24
        self.goal_rate = 3
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/'
        self.collect_data_srv = rospy.Service('/collect_data', Trigger, self.collect_data)
        self.goal_pub_srv = rospy.Service('/goal_pub', Trigger, self.goal_pub)
        self.save_img_no = 0
        self.clear_no = 0       
        self.offset_ang = 0
        # self.csv_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/'
        self.pos_list = []
        self.goal_list = []
        self.cur_pos = []
        self.pos = PoseWithCovarianceStamped()
        self.g_pos = PoseStamped()
        self.orientation = 0
        self.r = rospy.Rate(10)
        self.capture_rate = rospy.Rate(0.5)
        # self.capture_rate = rospy.Rate(0.25)
        # self.capture_rate = rospy.Rate(0.15)
        rospy.wait_for_service('/gazebo/set_model_state')
        self.state = ModelState()
        self.state.model_name = 'mobile_base'
        self.amcl_pose_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=1)
        self.simple_goal_pub = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10)
        os.makedirs(self.path + "img/" + self.name, exist_ok=True)
        os.makedirs(self.path + "ang/" + self.name, exist_ok=True)
        self.dl = deep_learning(n_action=1)
        

        with open(self.path + 'capture_pos/'+str(self.name)+'.csv', 'r') as fs:
        # with open(self.csv_path + 'capture_pos_fix.csv', 'r') as fs:
            for row in fs:
                self.pos_list.append(row)

    def capture_img(self):
            Flag = True
            try:
                cv2.imwrite(self.path + "/img/" + self.name + "/center" + str(self.save_img_no) + "_" + self.ang_no + ".jpg", self.im_resized)
                cv2.imwrite(self.path + "/img/" + self.name + "/right" + str(self.save_img_no) + "_" + self.ang_no + ".jpg", self.im_right_resized)
                cv2.imwrite(self.path + "/img/" + self.name + "/left" + str(self.save_img_no) + "_" + self.ang_no + ".jpg", self.im_left_resized)
            except:
                print('Not save image')
                Flag = False
            finally:
                if Flag:
                    print('Save image Number:', self.save_img_no)

    def capture_ang(self):
            line = [str(self.save_img_no), str(self.action)]
            with open(self.path + "ang/" + self.name + "/ang.csv" , 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
    
    def read_csv(self):
            self.cur_pos = self.pos_list[self.save_img_no]
            pos = self.cur_pos.split(',')
            x = float(pos[1])
            y = float(pos[2])
            theta = float(pos[3])
            return x, y, theta

    def simple_goal(self):
        list_num = self.save_img_no + self.goal_offset
        if list_num <= len(self.pos_list):
            self.cur_pos = self.pos_list[list_num]
            simple_pos = self.cur_pos.split(',')
            x = float(simple_pos[1])
            y = float(simple_pos[2])

            self.g_pos.header.stamp = rospy.Time.now()

            self.g_pos.header.frame_id = 'map'
            # self.g_pos.pose.position.x = x 
            # self.g_pos.pose.position.y = y
            #willow#
            self.g_pos.pose.position.x = x - 11.252
            self.g_pos.pose.position.y = y - 16.70
            self.g_pos.pose.position.z = 0

            self.g_pos.pose.orientation.x = 0 
            self.g_pos.pose.orientation.y = 0
            self.g_pos.pose.orientation.z = 0
            # self.g_pos.pose.orientation.w = 0.999
            self.g_pos.pose.orientation.w = 1.001

            self.simple_goal_pub.publish(self.g_pos)
        else:
            pass

    def robot_moving(self, x, y, angle):
            #amcl
            #replace_pose = PoseWithCovarianceStamped()

            self.pos.header.stamp = rospy.Time.now()

            self.pos.header.frame_id = 'map'
            #tsudanuma2-3#
            # self.pos.pose.pose.position.x = x
            # self.pos.pose.pose.position.y = y
            #willow#
            self.pos.pose.pose.position.x = x - 11.252
            self.pos.pose.pose.position.y = y - 16.70

            quaternion_ = tf.transformations.quaternion_from_euler(0, 0, angle)

            self.pos.pose.pose.orientation.x = quaternion_[0]
            self.pos.pose.pose.orientation.y = quaternion_[1]
            self.pos.pose.pose.orientation.z = quaternion_[2]
            self.pos.pose.pose.orientation.w = quaternion_[3]
            self.pos.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]
            
            # self.pos.pose.covariance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]

            # self.amcl_pose_pub.publish(self.pos)
            #gazebo
            # for self.offset_ang in [-7, -5, -3, 0, 3, 5, 7]:
            for self.offset_ang in [-5, 0, 5]:
                the = angle + math.radians(self.offset_ang)
                the = the - 2.0 * math.pi if the >  math.pi else the
                the = the + 2.0 * math.pi if the < -math.pi else the
                self.state.pose.position.x = x
                self.state.pose.position.y = y
                quaternion = tf.transformations.quaternion_from_euler(0, 0, the)
                self.state.pose.orientation.x = quaternion[0]
                self.state.pose.orientation.y = quaternion[1]
                self.state.pose.orientation.z = quaternion[2]
                self.state.pose.orientation.w = quaternion[3]

                # if self.offset_ang == -7:
                #     self.ang_no = "-7"
                
                if self.offset_ang == -5:
                    self.ang_no = "-5"

                # if self.offset_ang == -3:
                #     self.ang_no = "-3"

                if self.offset_ang == 0:
                    self.ang_no = "0"

                # if self.offset_ang == +3:
                #     self.ang_no = "+3"

                if self.offset_ang == +5:
                    self.ang_no = "+5"

                # if self.offset_ang == +7:
                #     self.ang_no = "+7"

                try:
                    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                    resp = set_state( self.state )

                    # self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                    self.im_resized = cv2.resize(self.cv_image, dsize=(64, 48))
                    # self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                    self.im_right_resized = cv2.resize(self.cv_right_image, dsize=(64, 48))
                    # self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                    self.im_left_resized = cv2.resize(self.cv_left_image, dsize=(64, 48))
                    
                    # if self.offset_ang == 0 and self.save_img_no % 3 == 0:
                    #     self.simple_goal()
                    # if self.offset_ang == -5:
                    #     self.amcl_pose_pub.publish(self.pos)

                    if self.offset_ang == 0 and self.save_img_no % self.goal_rate == 0:
                        self.simple_goal()
                    if self.save_img_no % 3 != 0:
                        self.capture_img()
                        self.capture_ang()
                    if self.offset_ang == -5:
                        self.amcl_pose_pub.publish(self.pos)
                    #test
                    # self.capture_img()
                    # self.capture_ang()
                except rospy.ServiceException as e:
                    print("Service call failed: %s" % e)
                self.r.sleep()
                self.r.sleep()
                self.r.sleep()
            
            self.r.sleep()
            self.r.sleep()
            self.r.sleep()
        

    def goal_pub(self):
        rospy.wait_for_service('/goal_pub')
        service = rospy.ServiceProxy('/goal_pub', Trigger)
        self.simple_goal()
    
    def collect_data(self, data):
        rospy.wait_for_service('/collect_data')
        service = rospy.ServiceProxy('/collect_data', Trigger)
        self.goal_pub()

        for i in range(len(self.pos_list)):
            x, y, theta = self.read_csv()
            self.robot_moving(x, y, theta)
            self.save_img_no += 1
            # print("current_position:", x, y, theta)
            # self.clear_no += 1
            # print("clear_no", self.clear_no)
            self.capture_rate.sleep()

            ##dist 0.1 dy 0.1 ##
            # if i == len(self.pos_list) - 11:
            ## dist 0.25 ##
            # if i == len(self.pos_list) - 18:
            ##dist 0.25 dy 0.05 ##
            # if i == len(self.pos_list) - 59:
            if i == len(self.pos_list):
                # for j in range(4000):
                #     self.dl.trains()
                # self.dl.save("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/")
                os.system('killall roslaunch')
                sys.exit()

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z
        

if __name__ == '__main__':
    rg = cource_following_learning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        # rg.loop()
        r.sleep() 
