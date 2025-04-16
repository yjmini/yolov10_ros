cmake_minimum_required(VERSION 3.0.2)
project(yolov10_ros_msgs)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  std_msgs
  message_generation
)

add_message_files(
        DIRECTORY msg
        FILES
        BoundingBox.msg
        BoundingBoxes.msg
        PersonMarkerData.msg
)

generate_messages(
        DEPENDENCIES
        geometry_msgs std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
