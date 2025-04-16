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
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from yolov10_ros_msgs.msg import PersonMarkerData

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
        weight_path = rospy.get_param('~weight_path', '/home/user/yolov10m.pt')
        sub_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov10/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        conf = float(rospy.get_param('~conf', '0.5'))
        self.visualize = rospy.get_param('~visualize', True)
        self.last_saved_time = 0
        self.save_interval = 7

        self.bridge = CvBridge()
        self.latest_depth = None
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)

        #self.person_pub = rospy.Publisher("/person_detected", Point, queue_size=10)
        self.person_marker_pub = rospy.Publisher("/person_marker_data", PersonMarkerData, queue_size=10) # Publisher

        self.device = 'cpu' if rospy.get_param('/use_cpu', 'false') else 'cuda'

        self.model = YOLO(weight_path)
        self.model.conf = conf

        self.color_image = Image()
        self.getImageStatus = False

        self.classes_colors = {}

        self.image_sub = rospy.Subscriber(sub_topic, Image, self.image_callback)

        self.position_pub = rospy.Publisher(pub_topic, BoundingBoxes, queue_size=1)
        self.image_pub = rospy.Publisher('/yolov10/detection_image', Image, queue_size=1)

        while not self.getImageStatus and not rospy.is_shutdown():
            rospy.loginfo("Waiting for image...")
            rospy.sleep(1)

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr("Depth 이미지 변환 실패: %s", str(e))

    def image_callback(self, image):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True

        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)
        self.color_image = cv2.resize(self.color_image, (640, 480))
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(self.color_image, show=False, conf=0.8)

        self.dectshow(results, image.height, image.width)

        cv2.waitKey(3)

    def dectshow(self, results, height, width):
        self.frame = self.color_image.copy()
        fps = 1000.0 / results[0].speed['inference']
        fps_int = int(fps)
        cv2.putText(self.frame, f'FPS: {fps_int}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

        person_detected_in_frame = False

        for result in results[0].boxes:
            cls_id = int(result.cls.item()) if hasattr(result.cls, "item") else int(result.cls)

            x1 = int(result.xyxy[0][0].item())
            y1 = int(result.xyxy[0][1].item())
            x2 = int(result.xyxy[0][2].item())
            y2 = int(result.xyxy[0][3].item())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cls_id == 0: # 사람이 감지되면
                person_detected_in_frame = True
                current_time = time.time()

                # 저장 간격이 되었는지 확인하고, 이미지 저장 및 좌표 발행
                if current_time - self.last_saved_time >= self.save_interval:
                    save_path = "/home/user/person_captures"
                    os.makedirs(save_path, exist_ok=True)

                    # 다음 저장될 파일 인덱스 계산
                    existing_files = [f for f in os.listdir(save_path) if f.startswith("person_") and f.endswith(".jpg")]
                    numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
                    next_index = max(numbers) + 1 if numbers else 1

                    save_name = os.path.join(save_path, f"person_{next_index}.jpg")

                    # 이미지 저장 시도
                    try:
                        cv2.imwrite(save_name, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)) # BGR로 변환하여 저장
                        rospy.loginfo(f"[YOLOv10] 사람 캡처 저장: {save_name}")
                        self.last_saved_time = current_time

                        # --- 중요: 저장 성공 후 좌표와 인덱스를 함께 발행 ---
                        if self.latest_depth is not None and 0 <= cx < self.latest_depth.shape[1] and 0 <= cy < self.latest_depth.shape[0]:
                            depth = self.latest_depth[cy, cx]
                            if depth > 0 and not np.isnan(depth) and not np.isinf(depth):
                                z = float(depth) / 1000.0
                                fx, fy = 384.4789, 384.4789
                                cx_d, cy_d = 319.7523, 245.3222
                                x = (cx - cx_d) * z / fx
                                y = (cy - cy_d) * z / fy

                                marker_data = PersonMarkerData()
                                marker_data.position.x = x
                                marker_data.position.y = y
                                marker_data.position.z = z
                                marker_data.image_index = next_index # 저장된 이미지 인덱스 포함

                                self.person_marker_pub.publish(marker_data) # 새 토픽으로 발행
                                rospy.loginfo(f"[YOLOv10] 마커 데이터 발행: index={next_index}, x={x:.2f}, y={y:.2f}, z={z:.2f}")
                        # --- 발행 로직 끝 ---

                    except Exception as e:
                        rospy.logerr(f"이미지 저장 실패: {save_name}, 오류: {e}")


            boundingBox = BoundingBox()
            boundingBox.xmin = x1
            boundingBox.ymin = y1
            boundingBox.xmax = x2
            boundingBox.ymax = y2
            boundingBox.Class = "person"
            boundingBox.probability = result.conf.item()
            self.boundingBoxes.bounding_boxes.append(boundingBox)

            label = f"person {boundingBox.probability:.2f}"
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.frame, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)

        self.position_pub.publish(self.boundingBoxes)
        self.publish_image(self.frame, height, width)

        if self.visualize:
            cv2.imshow('YOLOv10', self.frame)


    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.header = header
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)

if __name__ == "__main__":
    rospy.init_node('yolov10_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()