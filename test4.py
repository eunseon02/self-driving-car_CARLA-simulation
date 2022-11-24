#!/usr/bin/env python3
from tarfile import CompressionError
import rospy
import ros_numpy
#from carla_ros_bridge.sensor import Sensor, create_cloud
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField, Imu
from cv_bridge import CvBridge
from std_msgs.msg import Float32

import cv2                   
import matplotlib.pyplot as plt

import carla
import random
import torch
import keyboard
import numpy as np
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaWorldInfo
#from PIL import Image
import PIL
# from YOLOP.tools.demo2 import get_yoloP
from torchvision import transforms
from YOLOP.tools.demo import get_YOLOP

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
        
        # path = '/home/eunseon/baram_ws/src/test/src/images/simu'+ str(self.num) + '.jpg'
        
        
        # #resize = transforms.Resize((1280, 720))

        # #-----
        # #Image.fromarray() _ numpy->PIL
        # #resize
        # print(self.cv_image.shape) 
        # self.cv_image = cv2.resize(self.cv_image, (1280, 720))
        # print(self.cv_image.shape)
        # # resized_cv_image = resize(PIL.Image.fromarray((self.cv_image)))
        # # print(resized_cv_image.shape)
        # # print(type(resized_cv_image))

        # normalize = transforms.Normalize(
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # )
        # transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        # input= transform(self.cv_image)
        # print(input.shape)
        # output = model(input)
        # print(type(output))


        # output = model(torch.Tensor(resize(self.cv_image)))
        # res = Image.fromarray(output) #numpy -> PIL
        # model (res)

        # transform = transforms.ToTensor()
        # output = model(transform(self.cv_image))
        self.cv_image = cv2.resize(self.cv_image, (1280,720))
        cv2.imwrite(self.path, self.cv_image)
        # cv2.imshow('img', self.cv_image)
        
        # remove(self.path)

        # cv2.waitKey(1)
        # self.num+=1

class driving:
    def __init__(self):
        self.res_path = '/home/eunseon/baram_ws/src/test/src/inference/output/0.jpg'
        self.vel = 0.3
        self.str = 0.0
    def set(self):
        while not rospy.is_shutdown():
            self.topic_name_cmd = "/carla/ego_vehicle/vehicle_control_cmd"
       
            obj=CarlaEgoVehicleControl()
            obj.throttle = self.vel
            obj.steer = self.str
            self.driving_pub = rospy.Publisher(self.topic_name_cmd, CarlaEgoVehicleControl)
            self.driving_pub.publish(obj)
            
            #get_YOLOP()
            count, turn_right, turn_left, center_po = get_YOLOP()
            # width = 1280
            half_width = 640
            print(f"count : {count}")
            if(count > 150000):
                self.vel = 0.0
                self.str=0.0

            adaptive_weight = abs(half_width - center_po)/(half_width)
            own_weight = 1.5
            if(turn_right==1 and turn_left==0):
                self.str = adaptive_weight * own_weight if adaptive_weight * own_weight <= 1 else 1
                print(self.str)
                print('test4:turn right')
                #(right_count/count)*1
            elif(turn_right==0 and turn_left==1):
                self.str = adaptive_weight * -own_weight if adaptive_weight * -own_weight >= -1 else -1
                #(left_count/count)*(-1)
                print(self.str)
                print('test4:turn left')
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
    rospy.init_node("test", anonymous=True)
    #lidar()
    save_image()
    _driving = driving()
    _driving.set()

    rospy.spin()
    key = cv2.waitKey()
    if key == ord('q'):
        cv2.destroyAllWindows()