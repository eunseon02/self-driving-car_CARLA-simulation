#!/usr/bin/env python3
from tarfile import CompressionError
import rospy
import ros_numpy
#from carla_ros_bridge.sensor import Sensor, create_cloud
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Header

import cv2                   
import matplotlib.pyplot as plt

import carla
import random
import torch
import struct
import keyboard
import time
import numpy as np
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaWorldInfo
#from PIL import Image
import PIL
# from YOLOP.tools.demo2 import get_yoloP
from torchvision import transforms
from YOLOP.tools.demo2 import get_YOLOP

# model = get_yoloP()

class save_image():
    
    def __init__(self):
        self.image_sub = rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, self.callback)
        self.bridge = CvBridge()
        self.num = 0
        self.cv_image = None
        self.path = '/home/eunseon/baram_ws/src/test/src/images/simu/0.jpg'
        
    
    def callback(self, img):
        self.cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding = "bgr8")
        self.cv_image = cv2.resize(self.cv_image, (1280,720))
        cv2.imwrite(self.path, self.cv_image)

class check_initial_pos:
    def __init__(self):
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.callback)
        self.bridge = CvBridge()
        self.x = None
    def callback(self, data):
        #print(ros_numpy.geometry.(data))
        self.x = data.pose.pose.position.x
        # print(data)
        
class initial_position:
    def __init__(self):
        pass
    def set(self):
        #while not rospy.is_shutdown():
            self.topic_name_cmd = '/initialpose'
            #self.topic_name_cmd = '/carla/ego_vehicle/control/set_transform'
            initpose = PoseWithCovarianceStamped()
            initpose.pose.pose.position.x = -20
            initpose.pose.pose.position.y = 60.9
            initpose.pose.pose.orientation.x = 0
            initpose.pose.pose.orientation.y = 0
            initpose.pose.pose.orientation.z = 0
            initpose.pose.pose.orientation.w = 0.7071068

            self.initial_pose_pub = rospy.Publisher(self.topic_name_cmd, PoseWithCovarianceStamped)
            self.initial_pose_pub.publish(initpose)
        
            
class driving:
    def __init__(self):
        self.res_path = '/home/eunseon/baram_ws/src/test/src/inference/output/0.jpg'
        self.vel = 0.3
        self.str = 0.0
        self._intial_pose_set = initial_position()
        self.flag = 0
    def set(self):
        while not rospy.is_shutdown():
            if self.flag != 5:
                self._intial_pose_set.set()
                self.flag += 1

            self.topic_name_cmd = "/carla/ego_vehicle/vehicle_control_cmd"
       
            obj=CarlaEgoVehicleControl()
            obj.throttle = self.vel
            obj.steer = self.str
            self.driving_pub = rospy.Publisher(self.topic_name_cmd, CarlaEgoVehicleControl)
            self.driving_pub.publish(obj)
            
            #get_YOLOP()
            turn_right, turn_left, center_po = get_YOLOP()
            
            # width = 1280
            half_width = 640
            #print(f"count : {count}")
            # if(count > 150000):
            #     self.vel = 0.0
            #     self.str=0.0
            adaptive_weight = abs(half_width - center_po)/(half_width)
            own_weight = 0.20
            if(turn_right==0 and turn_left==1):
                time.sleep(0.049)
                self.str = adaptive_weight * own_weight if adaptive_weight * own_weight <= 1 else 1
                print('str = ', self.str)
                print('test4:turn right')
                #(right_count/count)*1
            elif(turn_right==1 and turn_left==0):
                time.sleep(0.049)
                self.str = adaptive_weight * -own_weight if adaptive_weight * -own_weight >= -1 else -1
                #(left_count/count)*(-1)
                print('str = ', self.str)
                print('test4:turn left')
            elif(turn_left ==1 and turn_right==1):
                self.str = 0
            elif(turn_right==0 and turn_left==0):
                self.vel=0 
            else:
                self.str = 0

class lidar:
    def __init__(self):
        self.lidar_pub = rospy.Subscriber("/carla/ego_vehicle/lidar", PointCloud2, self.sensor_data)
        self.bridge = CvBridge()

    def sensor_data(self, lidar):
        """for point in PointCloud2.read_points(lidar, skip_nans=True):
            pt_x = point[0]
            pt_y = point[1]
            pt_z = point[2]
        print(pt_x)"""
        print(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar))
         
class map:
    def __init__(self):
        while not rospy.is_shutdown():
            CarlaWorldInfo.map_name = 'Town10'
            self.map_pub = rospy.Publisher("/carla/world_info", CarlaWorldInfo.map_name)

if __name__ == '__main__':
    #map()
    rospy.init_node("test4", anonymous=True)
    check_initial_pos()
    e = initial_position()
    e.set()
    #lidar()
    save_image()
    _driving = driving()
    _driving.set()

    rospy.spin()
    key = cv2.waitKey()
    if key == ord('q'):
        cv2.destroyAllWindows()