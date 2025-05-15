#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import rospy
import numpy as np
import time
import tf2_ros
import tf2_geometry_msgs

from ultralytics import YOLO

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov10_ros_msgs.msg import BoundingBox, BoundingBoxes
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from yolov10_ros_msgs.msg import PersonMarkerData


class Yolo_Dect:
    def __init__(self):
        weight_path = rospy.get_param('~weight_path', '/home/user/yolov10m.pt')
        sub_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov10/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.tf_target_frame = rospy.get_param('~tf_target_frame', 'map')  # TF 좌표계 기준 (예: 'map')
        conf = float(rospy.get_param('~conf', '0.5'))
        self.visualize = rospy.get_param('~visualize', True)
        self.last_saved_time = 0
        self.save_interval = 7

        self.bridge = CvBridge()

        self.person_marker_pub = rospy.Publisher("/person_marker_data", PersonMarkerData, queue_size=10)

        self.device = 'cpu' if rospy.get_param('/use_cpu', 'false') else 'cuda'

        self.model = YOLO(weight_path)
        self.model.conf = conf

        self.color_image = None
        self.getImageStatus = False

        self.image_sub = rospy.Subscriber(sub_topic, Image, self.image_callback)

        self.position_pub = rospy.Publisher(pub_topic, BoundingBoxes, queue_size=1)
        self.image_pub = rospy.Publisher('/yolov10/detection_image', Image, queue_size=1)

        # TF listener 준비
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        while not self.getImageStatus and not rospy.is_shutdown():
            rospy.loginfo("Waiting for image...")
            rospy.sleep(1)

    def get_camera_pose(self):
        try:
            # 현재 시간 기준으로 카메라 프레임 -> TF 타겟 프레임 변환 정보 요청
            trans = self.tf_buffer.lookup_transform(self.tf_target_frame,
                                                    self.camera_frame,
                                                    rospy.Time(0),
                                                    rospy.Duration(1.0))
            # 변환 결과에서 위치 좌표 추출
            pos = trans.transform.translation
            return (pos.x, pos.y, pos.z)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF 변환 실패: {e}")
            return None

    def image_callback(self, image):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True

        self.color_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        self.color_image = cv2.resize(self.color_image, (640, 480))
        rgb_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(rgb_image, show=False, conf=0.8)

        self.dectshow(results, image.height, image.width, image.header)

        cv2.waitKey(3)

    def dectshow(self, results, height, width, image_header):
        self.frame = self.color_image.copy()
        fps = 1000.0 / results[0].speed['inference']
        cv2.putText(self.frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image_header
        self.boundingBoxes.image_header = image_header

        should_save_this_frame = False
        save_file_name = ""
        triggering_person_index = -1

        for result in results[0].boxes:
            cls_id = int(result.cls.item()) if hasattr(result.cls, "item") else int(result.cls)

            if cls_id != 0:
                continue  # 사람만 처리

            x1 = int(result.xyxy[0][0].item())
            y1 = int(result.xyxy[0][1].item())
            x2 = int(result.xyxy[0][2].item())
            y2 = int(result.xyxy[0][3].item())

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
            cv2.putText(self.frame, label, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            current_time = time.time()
            if not should_save_this_frame and current_time - self.last_saved_time >= self.save_interval:
                should_save_this_frame = True
                self.last_saved_time = current_time

                save_path = "/home/user/person_captures"
                os.makedirs(save_path, exist_ok=True)
                existing_files = [f for f in os.listdir(save_path) if f.startswith("person_") and f.endswith(".jpg")]
                numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
                next_index = max(numbers) + 1 if numbers else 1
                save_file_name = os.path.join(save_path, f"person_{next_index}.jpg")
                triggering_person_index = next_index

        if should_save_this_frame and save_file_name:
            try:
                cv2.imwrite(save_file_name, self.frame)
                rospy.loginfo(f"[YOLOv10] 사람(들) 포함 캡처 저장: {save_file_name}")

                # TF 좌표 가져오기
                camera_pos = self.get_camera_pose()
                if camera_pos is not None:
                    marker_data = PersonMarkerData()
                    marker_data.position.x = camera_pos[0]
                    marker_data.position.y = camera_pos[1]
                    marker_data.position.z = camera_pos[2]
                    marker_data.image_index = triggering_person_index
                    self.person_marker_pub.publish(marker_data)
                    rospy.loginfo(f"[YOLOv10] 마커 데이터 발행 (TF 좌표): index={triggering_person_index}, "
                                  f"x={marker_data.position.x:.2f}, y={marker_data.position.y:.2f}, z={marker_data.position.z:.2f}")
                else:
                    rospy.logwarn("TF 좌표를 받지 못해 마커 데이터 발행 실패")

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
