cmake_minimum_required(VERSION 3.0)
project(visiondiffusiontesting)

SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_SKIP_RPATH False)
SET(CMAKE_INSTALL_RPATH ..)

find_package(catkin REQUIRED COMPONENTS roscpp rospy roslib tf2_ros geometry_msgs sensor_msgs std_msgs genmsg)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS message_runtime
)

catkin_package()
find_package(Eigen3 REQUIRED)


