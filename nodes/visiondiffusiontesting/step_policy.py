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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from robotreset import gotoPose
from scipy.spatial.transform import Rotation as ScipyR

class TrajectoryVisualizer:
    def __init__(self, topic_name="/forecasted_trajectory", frame_id="base"):
        self.publisher = rospy.Publisher(topic_name, MarkerArray, queue_size=1)
        self.frame_id = frame_id

    def publish_trajectory(self, trajectory: np.ndarray, color=(0.0, 1.0, 0.0, 0.8)):
        """
        Publishes a trajectory as a MarkerArray in RViz.

        :param trajectory: np.ndarray of shape (T, 3) representing [x, y, z] positions
        :param color: tuple of (r, g, b, a)
        """
        marker_array = MarkerArray()
        t_now = rospy.Time.now()

        if len(np.shape(trajectory))==3:
            trajectory = trajectory.reshape(-1, trajectory.shape[-1])

        print(np.shape(trajectory))

        for i, point in enumerate(trajectory):
            marker = Marker()
            marker.header.stamp = t_now
            marker.header.frame_id = self.frame_id
            marker.ns = "trajectory"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.002
            marker.scale.y = 0.002
            marker.scale.z = 0.002

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            marker.lifetime = rospy.Duration(0)  # doesn't fade
            marker_array.markers.append(marker)

        self.publisher.publish(marker_array)

def plot_3d_trajectories(data: np.ndarray, title: str = "3D Trajectory"):
    """
    Plot a single 3D trajectory from a 32x7 array using Plotly.

    Args:
        data (np.ndarray): A 32x7 array where the first three columns are x, y, z coordinates.
        title (str): Plot title.
    """
    if data.shape[1] < 3:
        raise ValueError("Input data must have at least 3 columns for x, y, z.")

    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(size=3),
            line=dict(width=2)
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title=title
    )

    fig.show()

class StepPolicy:
    def __init__(self, model_path):
        self.policy = VisionDiffusionUNet()
        self.policy.load(model_path)
        rospy.loginfo(f"[step_policy] Loaded diffusion policy from {model_path}")

        rospy.on_shutdown(self.on_shutdown)

        self.current_pose = None
        self.current_image = None
        self.last_action = None
        self.last_cmd = None
        self.bridge = CvBridge()

        self.visualizer = TrajectoryVisualizer()

        # for homing
        self.home_pos = np.array([0.275, -0.343, 0.293])
        self.home_quat = [0.10318,0.0925,0.7008,-0.6997] #xyzw
        self.home_quat_R = ScipyR.from_quat(self.home_quat)

        self.chunking_index = 0
        self.max_chunk = 31

        rospy.Subscriber('/ur5e/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/robot_webcam/image_raw', Image, self.image_cb)
        self.cmd_pub = rospy.Publisher('/ur5e/compliant_pose', PoseStamped, queue_size=1)
        self.auton_pub = rospy.Publisher('/activate_autonomy', Bool, queue_size=1)

        time.sleep(1)

        self.auton_pub.publish(Bool(True))

        self.fd = sys.stdin.fileno()
        self.old_term = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

        print("[step_policy] Ready. Waiting for pose and image...")

    def on_shutdown(self):
        self.auton_pub.publish(Bool(False))


    def pose_cb(self, msg: PoseStamped):
        self.current_pose = msg

    def image_cb(self, msg: Image):
        self.current_image = msg

    def compute_action(self):
        if self.current_pose is None or self.current_image is None:
            rospy.logwarn("[step_policy] No pose or image received yet.")
            return

        p = self.current_pose.pose
        state_vec = np.array([
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
        ], dtype=np.float32)

        img_np = self.bridge.imgmsg_to_cv2(self.current_image, desired_encoding='rgb8')

        # try:
        if self.chunking_index==0:
            # print("SV:",state_vec)
            self.action = self.policy.getAction(state_vec, img_np, forecast=True)
            self.actions = []
            for ii in range(25):
                self.actions.append(copy.deepcopy(self.policy.getAction(state_vec, img_np, forecast=True)))

        print("for valerie:",np.shape(np.array(self.actions)))
        # self.visualizer.publish_trajectory(self.action[:,0:3])
        self.visualizer.publish_trajectory(np.array(self.actions)[:,:,0:3])

        # print(np.shape(self.action))
        # plot_3d_trajectories(self.action)

        action = self.action[self.chunking_index,:]
      
        # except Exception as e:
        #     rospy.logerr(f"[step_policy] policy.getAction failed: {e}")
        #     return

        print("[step_policy] Current State:", state_vec)
        print("[step_policy] Proposed action:", action)
        print("[step_policy] Delta:", action-state_vec)
        print("ci:",self.chunking_index)

        cmd = PoseStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = self.current_pose.header.frame_id
        cmd.pose.position.x = float(action[0])
        cmd.pose.position.y = float(action[1])
        cmd.pose.position.z = float(action[2])
        cmd.pose.orientation.x = float(action[3])
        cmd.pose.orientation.y = float(action[4])
        cmd.pose.orientation.z = float(action[5])
        cmd.pose.orientation.w = float(action[6])

        self.last_action = action
        self.last_cmd = cmd

        self.chunking_index+=1
        if self.chunking_index > self.max_chunk:
            self.chunking_index = 0

    def run(self):
        try:
            while not rospy.is_shutdown():
                # Wait for new inputs and compute action
                self.compute_action()

                # Wait for keypress to confirm execution
                print("[step_policy] Press 's' to execute action...")
                while not rospy.is_shutdown():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        c = sys.stdin.read(1)
                        if c.lower() == 's':
                            if self.last_cmd:
                                self.cmd_pub.publish(self.last_cmd)
                                print("[step_policy] Action executed.")
                            else:
                                print("[step_policy] No command to send.")
                            break
                        elif c.lower() == 'p':
                            rand_rot = ScipyR.from_rotvec(np.random.normal(0,0.015,3))
                            new_quat = (self.home_quat_R*rand_rot).as_quat()
                            pos_noise = np.random.normal(0,0.003,3)
                            pos = copy.deepcopy(self.home_pos) + pos_noise
                            gotoPose(pos,new_quat)
                            self.chunking_index = 0
                            rospy.sleep(0.01) 
                            print("Resetting!")
                            break
                    rospy.sleep(0.01) 
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_term)
            rospy.loginfo("[step_policy] Terminal restored.")



if __name__ == "__main__":
    import os

    # You can modify this path as needed
    DEFAULT_MODEL_PATH = os.path.expanduser(project_path+'/../../policies/visiondiffusion_7-07_obs_hor_2_pred_32.pkl')
    # visiondiffusion_ur5e_5-30
    # visiondiffusion_ur5e_6-3_obs_hor_4_pred_16

    parser = argparse.ArgumentParser(description='Step through policy with confirmation before execution')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the trained .pkl model file (default: ~/models/diffusion_policy.pkl)')
    args = parser.parse_args()

    rospy.init_node('step_policy', anonymous=False)
    executor = StepPolicy(args.model_path)
    executor.run()
