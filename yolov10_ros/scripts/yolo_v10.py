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
        cv2.putText(self.frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # 이번 프레임의 모든 사람 바운딩 박스를 담을 리스트 초기화
        self.boundingBoxes = BoundingBoxes()
        # --- 중요: header 설정 ---
        # image_callback에서 받은 image 메시지의 header를 사용하는 것이 좋습니다.
        # self.image_callback에서 self.image_header = image.header 와 같이 저장해두고 사용하거나,
        # image 메시지 자체를 dectshow로 넘겨받아 사용해야 합니다.
        # 아래는 임시 예시이며, 실제 image.header를 사용하도록 수정해야 할 수 있습니다.
        try:
            # image_callback에서 image 헤더를 self.last_image_header 등으로 저장했다고 가정
            self.boundingBoxes.header = self.last_image_header
            self.boundingBoxes.image_header = self.last_image_header
        except AttributeError:
            rospy.logwarn("Image header not found for BoundingBoxes message.")
            current_ros_time = rospy.Time.now()
            self.boundingBoxes.header.stamp = current_ros_time
            self.boundingBoxes.image_header.stamp = current_ros_time
            self.boundingBoxes.header.frame_id = self.camera_frame 
            self.boundingBoxes.image_header.frame_id = self.camera_frame 
        # ----------------------

        # 저장 및 발행 관련 변수 초기화
        should_save_this_frame = False  # 이번 프레임을 저장해야 하는지 여부
        save_file_name = ""             # 저장될 파일 이름
        triggering_person_coords = None # 저장을 유발한 사람의 좌표
        triggering_person_index = -1    # 저장될 이미지의 인덱스

        for result in results[0].boxes:
            cls_id = int(result.cls.item()) if hasattr(result.cls, "item") else int(result.cls)

            # 사람이 아니면 건너뛰기
            if cls_id != 0:
                continue

            # 사람(cls_id == 0)인 경우
            x1 = int(result.xyxy[0][0].item())
            y1 = int(result.xyxy[0][1].item())
            x2 = int(result.xyxy[0][2].item())
            y2 = int(result.xyxy[0][3].item())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            boundingBox = BoundingBox()
            boundingBox.xmin = x1
            boundingBox.ymin = y1
            boundingBox.xmax = x2
            boundingBox.ymax = y2
            boundingBox.Class = "person"
            boundingBox.probability = result.conf.item()
            self.boundingBoxes.bounding_boxes.append(boundingBox)

            # 바운딩 박스 그리기
            label = f"person {boundingBox.probability:.2f}"
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.frame, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)

            current_time = time.time()
            if not should_save_this_frame and current_time - self.last_saved_time >= self.save_interval:
                should_save_this_frame = True 
                self.last_saved_time = current_time 

                # 저장 파일 이름 및 인덱스 계산
                save_path = "/home/user/person_captures" # 경로
                os.makedirs(save_path, exist_ok=True)
                existing_files = [f for f in os.listdir(save_path) if f.startswith("person_") and f.endswith(".jpg")]
                numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
                next_index = max(numbers) + 1 if numbers else 1
                save_file_name = os.path.join(save_path, f"person_{next_index}.jpg")
                triggering_person_index = next_index # 인덱스 저장

                # 사람의 3D 좌표 계산 및 저장 (마커 발행)
                if self.latest_depth is not None and 0 <= cx < self.latest_depth.shape[1] and 0 <= cy < self.latest_depth.shape[0]:
                    depth = self.latest_depth[cy, cx]
                    if depth > 0 and not np.isnan(depth) and not np.isinf(depth):
                        z = float(depth) / 1000.0
                        # --- 카메라 파라미터---
                        fx, fy = 384.4789, 384.4789
                        cx_d, cy_d = 319.7523, 245.3222
                        # -------------------
                        x = (cx - cx_d) * z / fx
                        y = (cy - cy_d) * z / fy
                        triggering_person_coords = {'x': x, 'y': y, 'z': z} # 좌표 저장
                    else:
                        triggering_person_coords = None 
                        rospy.logwarn(f"저장 트리거: Depth 값({depth})이 유효하지 않아 좌표 계산 실패 (cx={cx}, cy={cy})")
                else:
                    triggering_person_coords = None 
                    rospy.logwarn("저장 트리거: Depth 이미지가 없거나 중심 좌표가 범위를 벗어나 좌표 계산 실패")

        # ===== 루프 종료 =====

        # 이미지 저장 및 마커 좌표 발행 
        if should_save_this_frame and save_file_name:
            try:
                # 모든 사람의 바운딩 박스가 그려진 최종 self.frame 저장 (RGB->BGR 변환)
                cv2.imwrite(save_file_name, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
                rospy.loginfo(f"[YOLOv10] 사람(들) 포함 캡처 저장: {save_file_name}")

                # 저장을 유발했던 사람의 좌표 데이터 발행
                if triggering_person_coords is not None and triggering_person_index != -1:
                    marker_data = PersonMarkerData()
                    marker_data.position.x = triggering_person_coords['x']
                    marker_data.position.y = triggering_person_coords['y']
                    marker_data.position.z = triggering_person_coords['z']
                    marker_data.image_index = triggering_person_index
                    self.person_marker_pub.publish(marker_data)
                    rospy.loginfo(f"[YOLOv10] 마커 데이터 발행 (트리거 기준): index={triggering_person_index}, "
                                  f"x={marker_data.position.x:.2f}, y={marker_data.position.y:.2f}, z={marker_data.position.z:.2f}")
                else:
                     rospy.logwarn(f"이미지({save_file_name}) 저장 후 마커 데이터 발행 실패: 저장된 좌표 없음")

            except Exception as e:
                rospy.logerr(f"최종 이미지 저장 실패: {save_file_name}, 오류: {e}")

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