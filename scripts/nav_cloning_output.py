#!/usr/bin/env python3
from __future__ import print_function
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_pytorch import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
#from gazebo_msgs.srv import SetModelState
#from gazebo_msgs.srv import GetModelState
#from gazebo_msgs.msg import ModelState
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
from nav_msgs.msg import Odometry

class cource_following_learning_node:
    def __init__(self):
        rospy.init_node('cource_following_learning_node', anonymous=True)
        self.action_num = rospy.get_param("/LiDAR_based_learning_node/action_num", 1)
        print("action_num: " + str(self.action_num))
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.learning = True
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model/'
        self.previous_reset_time = 0
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)
        self.load_path = ("/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/model/cit2-3_4000_1/model_gpu.pt")

        self.odom_sub = rospy.Subscriber("/tracker", Odometry, self.path_write)
        self.path_pose_x = 0
        self.path_pose_y = 0
        self.path_no = 0

        with open(self.path +  'analysis/trajectory/exp1.2/8000_1.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')

        with open(self.path + self.start_time + '/' +  'reward.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)'])

    def path_write(self, data):
            with open(self.path + 'analysis/trajectory/exp1.2/8000_1.csv', 'a') as f:
                self.path_pose_x = data.pose.pose.position.x
                self.path_pose_y = data.pose.pose.position.y
                path_line = [str(self.path_no), str(self.path_pose_x), str(self.path_pose_y)]
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(path_line)
            self.path_no += 1

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

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position

        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos.x - path.x)**2 + (pos.y - path.y)**2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        """
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        try:
            previous_model_state = get_model_state('mobile_base', 'world')
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        """
        
        # if self.vel.linear.x == 0:
        #     return

        img = resize(self.cv_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img)
        imgobj = np.asanyarray([r,g,b])

        img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img_left)
        imgobj_left = np.asanyarray([r,g,b])

        img_right = resize(self.cv_right_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img_right)
        imgobj_right = np.asanyarray([r,g,b])

        ros_time = str(rospy.Time.now())


        if self.episode == 0:
            self.learning = False
            #self.dl.save(self.save_path)
            self.dl.load(self.load_path)

        
        target_action = self.dl.act(img)
        distance = self.min_distance
        print("TEST MODE: " + " angular:" + str(target_action) + ", distance: " + str(distance))

        self.episode += 1
        angle_error = abs(self.action - target_action)
        line = [str(self.episode), "test", "0", str(angle_error), str(distance)]
        with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(line)
        self.vel.linear.x = 0.2
        self.vel.angular.z = target_action
        self.nav_pub.publish(self.vel)

        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        temp = copy.deepcopy(img_left)
        cv2.imshow("Resized Left Image", temp)
        temp = copy.deepcopy(img_right)
        cv2.imshow("Resized Right Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = cource_following_learning_node()
    DURATION = 0.5
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()