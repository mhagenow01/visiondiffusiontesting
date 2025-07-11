#!/usr/bin/env python3
import sys
import select
import termios
import tty
import argparse
import time
import copy

import os
project_path = os.path.dirname(os.path.realpath(__file__))


import rospy
import numpy as np
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_diffusion_unet import VisionDiffusionUNet

import numpy as np
import plotly.graph_objects as go

import rospy

import tf2_ros
from scipy.spatial.transform import Slerp




from scipy.spatial.transform import Rotation as ScipyR

def gotoPose(pos,quat):
    # max_vel = 0.03 # m/s
    # max_angvel = 0.2 # rad/s
    # vkchen edit for faster data collection
    max_vel = 0.15 # m/s
    max_angvel = 0.4 # rad/s   
    cmd_pub = rospy.Publisher('/ur5e/compliant_pose', PoseStamped, queue_size=1)



    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rate = rospy.Rate(10)

    auton_pub = rospy.Publisher('/activate_autonomy', Bool, queue_size=1)

    time.sleep(1)

    auton_pub.publish(Bool(True))


    curr_pos_acquired = False

    while not curr_pos_acquired:
        try:
            trans = tfBuffer.lookup_transform('base', 'toolnew', rospy.Time())
            curr_pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            curr_q = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            curr_pos_acquired = True
        except Exception as e:
            print(e)
        rate.sleep()

        # Step 1: go up
        des_z = pos[2]
        dist_tmp = np.linalg.norm(des_z-curr_pos[2])
        num_steps = max(1,int(10*(dist_tmp/max_vel))) # 5cm/s

        for tt in range(num_steps):
            pose_out = PoseStamped()
            pose_out.header.frame_id = 'map'
            pose_out.header.stamp = rospy.Time.now()
            pose_out.pose.position.x = curr_pos[0]
            pose_out.pose.position.y = curr_pos[1]
            pose_out.pose.position.z = curr_pos[2]+float(tt)/num_steps*(des_z-curr_pos[2])
            pose_out.pose.orientation.x = curr_q[0]
            pose_out.pose.orientation.y = curr_q[1]
            pose_out.pose.orientation.z = curr_q[2]
            pose_out.pose.orientation.w = curr_q[3]
            cmd_pub.publish(pose_out)
            rate.sleep()

        curr_pos[2] = des_z

        # Step 2: interpolate and rotate
        dist_tmp = np.linalg.norm(pos-curr_pos)
        num_steps_lin = max(1,int(10*(dist_tmp/max_vel))) # 10cm/s

        curr_q_sp = ScipyR.from_quat(curr_q)
        quat_sp = ScipyR.from_quat(quat)
        ang_dist = np.linalg.norm((curr_q_sp.inv() * quat_sp).as_rotvec())
        num_steps_ang = max(1,int(10*(ang_dist/max_angvel))) # 10cm/s

        num_steps = max(num_steps_lin,num_steps_ang)
        
        key_rots = ScipyR.from_quat([curr_q,quat])
        slerper = Slerp([0,num_steps],key_rots)

        for tt in range(num_steps):
            interp_pos = curr_pos + float(tt)/num_steps*(pos-curr_pos) # TODO: technically num_steps-1
            interp_q = slerper(tt).as_quat()
            pose_out = PoseStamped()
            pose_out.header.frame_id = 'map'
            pose_out.header.stamp = rospy.Time.now()
            pose_out.pose.position.x = interp_pos[0]
            pose_out.pose.position.y = interp_pos[1]
            pose_out.pose.position.z = interp_pos[2]
            pose_out.pose.orientation.x = interp_q[0]
            pose_out.pose.orientation.y = interp_q[1]
            pose_out.pose.orientation.z = interp_q[2]
            pose_out.pose.orientation.w = interp_q[3]
            cmd_pub.publish(pose_out)
            rate.sleep()


        time.sleep(1)
        auton_pub.publish(Bool(False))



if __name__ == "__main__":
    rospy.init_node('robot_reset', anonymous=False)


    pos = [0.27, -0.37, 0.25]
    quat = [0,0,0.7071,-0.7071] #xyzw

    # TODO: add noise

    gotoPose(pos,quat)
