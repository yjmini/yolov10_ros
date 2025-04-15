#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import torch
import rospy
import numpy as np
from ultralytics import YOLO
import time

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov10_ros_msgs.msg import BoundingBox, BoundingBoxes

COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


class Yolo_Dect:
    def __init__(self):

        # load parameters
        weight_path = rospy.get_param('~weight_path', '/home/user/yolov10m.pt')
        sub_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic'  , '/yolov10/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')
        self.visualize = rospy.get_param('~visualize', 'True')
        self.last_saved_time = 0  # 마지막으로 저장한 시각 (초)
        self.save_interval = 7    # 저장 주기 (초)

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        self.model = YOLO(weight_path)
        self.model.conf = conf
        self.color_image = Image()
        self.getImageStatus = False

        # Load class color
        self.classes_colors = {}

        # image subscribe
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)

        self.image_pub = rospy.Publisher(
            '/yolov10/detection_image',  Image, queue_size=1)

        # if no image messages
        while (not self.getImageStatus):
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def image_callback(self, image):

        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)

        # 영상 크기를 640x480으로 리사이즈
        self.color_image = cv2.resize(self.color_image, (640, 480))

        # 영상 색상 변환
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        # 사람 탐지
        results = self.model(self.color_image, show=False, conf=0.9)

        self.dectshow(results, image.height, image.width)

        cv2.waitKey(3)

    def dectshow(self, results, height, width):
        self.frame = self.color_image.copy()  # 원본 RGB 이미지를 기준으로 박스 그리기
        print(str(results[0].speed['inference']))
        fps = 1000.0 / results[0].speed['inference']
        fps_int = int(fps)
        cv2.putText(self.frame, f'FPS: {fps_int}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)

        person_detected = False  # 사람 인식 여부 플래그

        for result in results[0].boxes:
            cls_id = int(result.cls.item()) if hasattr(result.cls, "item") else int(result.cls)
            if cls_id != 0:
                continue  # 사람(class_id=0) 외에는 무시

            person_detected = True  # 사람 인식됨

            boundingBox = BoundingBox()
            boundingBox.xmin = np.int64(result.xyxy[0][0].item())
            boundingBox.ymin = np.int64(result.xyxy[0][1].item())
            boundingBox.xmax = np.int64(result.xyxy[0][2].item())
            boundingBox.ymax = np.int64(result.xyxy[0][3].item())
            boundingBox.Class = "person"
            boundingBox.probability = result.conf.item()
            self.boundingBoxes.bounding_boxes.append(boundingBox)

            # 시각화 - 화면에 그리기
            x1, y1 = boundingBox.xmin, boundingBox.ymin
            x2, y2 = boundingBox.xmax, boundingBox.ymax
            conf = boundingBox.probability
            label = f"person {conf:.2f}"

            # 바운딩 박스 그리기
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 바운딩 박스 내부 왼쪽 위에 텍스트 표시
            cv2.putText(self.frame, label, (x1+5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)

        self.position_pub.publish(self.boundingBoxes)
        self.publish_image(self.frame, height, width)

        if self.visualize:
            cv2.imshow('YOLOv10', self.frame)
        
        # test
        # 사람 인식되었으면 이미지 저장
        if person_detected:
            current_time = time.time()
            if current_time - self.last_saved_time >= self.save_interval:
                save_path = "/home/user/person_captures"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # 기존 파일들 중 가장 큰 번호 찾기
                existing_files = [f for f in os.listdir(save_path) if f.startswith("person_") and f.endswith(".jpg")]
                numbers = [
                    int(f.split("_")[1].split(".")[0])
                    for f in existing_files
                    if f.split("_")[1].split(".")[0].isdigit()
                ]
                next_index = max(numbers) + 1 if numbers else 1

                # 파일 이름 저장
                save_name = os.path.join(save_path, f"person_{str(next_index).zfill(4)}.jpg")
                cv2.imwrite(save_name, self.frame)
                rospy.loginfo(f"[YOLOv10] 사람 캡처 저장: {save_name}")
                self.last_saved_time = current_time  # 마지막 저장 시각 갱신

    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)

if __name__ == "__main__":
    rospy.init_node('yolov10_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()
