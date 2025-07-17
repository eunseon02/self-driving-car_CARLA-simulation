#!/usr/bin/env python3
from tarfile import CompressionError
import rospy
import ros_numpy
#from carla_ros_bridge.sensor import Sensor, create_cloud
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
from std_msgs.msg import Float32

import cv2                   
import matplotlib.pyplot as plt

import carla
import random
import torch
import keyboard
import time
import numpy as np
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaWorldInfo
#from PIL import Image
import PIL
# from YOLOP.tools.demo2 import get_yoloP
from torchvision import transforms
from YOLOP.tools.demo4 import get_YOLOP

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
        # self.turn_right = 0
        # self.turn_left = 0 
        self.center_po = 640   


    def set_dir(self, img_binary):

        turn_left = 0
        turn_right = 0
       
        contours, _ = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        center_point = []
        for i in contours:
            area = cv2.contourArea(i)
            if area >10000:
                M = cv2.moments(i)
                #print(M)
                
                if M['m00'] != 0.0 and M['m00'] != 0.0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    center_point.append([cX, cY])
 
        lines_cnt = len(center_point)
       
        print('인식된 라인 수', lines_cnt)
        print('center point: ', center_point)
        lines = cv2.HoughLinesP(img_binary, 1, np.pi/180, 100, minLineLength = 40, maxLineGap = 5)
        if lines is None:
            turn_left = 1
            turn_right = 1
            # turn_left = 1
            # turn_right = 1
        else:
            for i in range(lines.shape[0]):
                point_1 = [lines[i][0][0], lines[i][0][1]] 
                point_2 = [lines[i][0][2], lines[i][0][3]] 
                dir = (point_2[1] - point_1[1])/(point_2[0] - point_1[0])
     
            if abs(point_1[0]-point_2[0])>1000:
                turn_left = 0
                turn_right = 0
                # turn_left = 0
                # turn_right = 0
            else:
                if lines_cnt<1:
                    # turn_left = 1
                    # turn_right = 1
                    turn_left = 1
                    turn_right = 1

                elif lines_cnt==1: 
                    cv2.circle(img_binary, (center_point[0][0], center_point[0][1]), 3, (0, 255, 255), -1)
                        #center_po = (one_line[i][0][0]+one_line[i][0][2])/2
                    
                    set = 1.0
                    self.center_po = (abs(640-(lines[i][0][0]+lines[i][0][2])/2)/640*set)*630+640
                    #center_po = str*640+640
                    print('point1 : ',lines[i][0][0],'point2 : ', lines[i][0][2])
                    print('dir = ', dir)
                    if dir<0:
                        print('one line : turn right')
                        turn_right = 1
                        # print('one line : turn left')
                        # turn_left = 1
                        
                    else:
                        # print('one line : turn right')
                        # turn_right = 1
                        print('one line : turn left')
                        turn_left = 1
                        

                elif lines_cnt==2:
                    po1 = center_point[0][0]
                    po2 = center_point[1][0]
                    cv2.circle(img_binary, (po1, center_point[0][1]), 3, (0, 0, 0), -1)
                    cv2.circle(img_binary, (po2, center_point[1][1]), 3, (0, 0, 0), -1)
                   
                    # 예외처리
                    if po1>600 and po2>600:
                        print('turn left')
                        turn_left = 1
                    elif po1<700 and po2<700:
                        print('turn right')
                        turn_right = 1
                    else:
                        # set = 0.8
                        # self.center_po = (abs(640-(lines[i][0][0]+lines[i][0][2])/2)/640*set)*630+640
                        self.center_po = (po1+po2)/2
                        if dir<0:
                            # print('two line : turn left')
                            # turn_left = 1
                            print('two line : turn right')
                            turn_right = 1
                        
                        else:
                            # print('two line : turn right')
                            # turn_right = 1
                            print('two line : turn left')
                            turn_left = 1
            
       
                            
#_______________________________
                else: # contour가 3개 이상 인식됨
                    po1 = 0
                    po2 = 0
                    print(center_point)
                    
                    for i in range (0, lines_cnt):  
                            cv2.circle(img_binary, (po1, center_point[0][1]), 3, (0, 0, 0), -1)
                            cv2.circle(img_binary, (po2, center_point[1][1]), 3, (0, 0, 0), -1)
                            print(po1, po2)
                            break

                    self.center_po = (po1+po2)/2
                    if dir<0:
                        print('multi line : turn right')
                        turn_right = 1
                    
                    else:
                        print('multi line : turn left')
                        self.turn_left = 1
                cv2.imshow('img_3', img_binary)
        return turn_left, turn_right                

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
            
            turn_left = 0
            turn_right = 0

            #turn_right, turn_left, center_po = get_YOLOP()
            img_binary = get_YOLOP()
            res_left, res_right = self.set_dir(img_binary)
            # cv2.imshow('img', img_binary)
        
            turn_left = res_left
            turn_right = res_right

            # width = 1280
            half_width = 640
       
            adaptive_weight = abs(half_width - self.center_po)/(half_width)
            own_weight = 0.20
            
            if(turn_right==1 and turn_left==0):
                time.sleep(0.02)
                self.str = adaptive_weight * own_weight if adaptive_weight * own_weight <= 1 else 1
                print('str = ', self.str)
                print('test4:turn right')
                #(right_count/count)*1
            elif(turn_right==0 and turn_left==1):
                time.sleep(0.02)
                self.str = adaptive_weight * -own_weight if adaptive_weight * -own_weight >= -1 else -1
                #(left_count/count)*(-1)
                print('str = ', self.str)
                print('test4:turn left')
            elif(turn_left ==1 and turn_right==1):
                self.str = 0
                print('test4:')
            elif(turn_right==0 and turn_left==0):
                #self.vel=0 
                print('test4:')
            else:
                self.str = 0
                print('test4:')


# class lidar:
#     def __init__(self):
#         self.lidar_pub = rospy.Subscriber("/carla/ego_vehicle/lidar", PointCloud2, self.sensor_data)
#         self.bridge = CvBridge()

#     def sensor_data(self, lidar):
#         """for point in PointCloud2.read_points(lidar, skip_nans=True):
#             pt_x = point[0]
#             pt_y = point[1]
#             pt_z = point[2]
#         print(pt_x)"""
#         print(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(lidar))
         
# class map:
#     def __init__(self):
#         while not rospy.is_shutdown():
#             CarlaWorldInfo.map_name = 'Town10'
#             self.map_pub = rospy.Publisher("/carla/world_info", CarlaWorldInfo.map_name)

if __name__ == '__main__':
    #map()
    rospy.init_node("test", anonymous=True)
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