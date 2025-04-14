# Yolov10_ros

This package provides a ROS wrapper for [ultralytics](https://github.com/ultralytics/ultralytics) based on PyTorch-YOLOv10. The package has been tested with Ubuntu 22.04.

# develop environment：
- Ubuntu 22.04
- ROS Noetic

# Prerequisites:
```
pip install ultralytics
pip install feh
```

## Installation
```
cd /your/catkin_ws/src
git clone https://github.com/yjmini/yolov10_ros.git
cd yolov10_ros
git submodule update --init --recursive
cd ..
rosdep update
rosdep install --from-paths src --ignore-src -r -y 
catkin_make
```

## Basic Usage
```
roslaunch yolov10_ros yolo_v10.launch
```

## detection images
```
cd ~/person_captures
feh person_0001.jpg
```

### Node parameters

* **`sub_topic`** 

    Subscribed camera topic.

* **`weights_path`** 

    Path to weights file.

* **`pub_topic`** 

    Published topic with the detected bounding boxes.
    
* **`confidence`** 

    Confidence threshold for detected objects.
    


