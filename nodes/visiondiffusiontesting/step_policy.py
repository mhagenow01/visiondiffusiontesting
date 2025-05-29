#!/usr/bin/env python3
import sys
import select
import termios
import tty
import argparse

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# adjust import to your policy module location
from policies.diffusion_with_images import Diffusion

class StepPolicy:
    def __init__(self, model_path):
        # Load the trained diffusion policy
        self.policy = Diffusion()
        self.policy.load(model_path)
        rospy.loginfo(f"[step_policy] Loaded diffusion policy from {model_path}")

        # Holders for latest observations
        self.current_pose = None
        self.current_image = None
        self.bridge = CvBridge()

        # ROS subscriptions
        rospy.Subscriber('/ur5e/compliant_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/robot_webcam/image_raw',  Image,       self.image_cb)

        # Set terminal to cbreak mode to capture single-key presses
        self.fd = sys.stdin.fileno()
        self.old_term = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

        print("[step_policy] Ready. Press 's' to step policy and print next action.")

    def pose_cb(self, msg: PoseStamped):
        self.current_pose = msg

    def image_cb(self, msg: Image):
        self.current_image = msg

    def step(self):
        # Ensure we have both pose and image
        if self.current_pose is None or self.current_image is None:
            rospy.logwarn("[step_policy] No pose or image received yet.")
            return

        # Build state vector (7-dim)
        p = self.current_pose.pose
        state_vec = np.array([
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
        ], dtype=np.float32)

        # Convert ROS Image -> RGB numpy
        img_np = self.bridge.imgmsg_to_cv2(self.current_image, desired_encoding='rgb8')

        # Compute next action
        try:
            action = self.policy.getAction(state_vec, img_np, forecast=False)
        except Exception as e:
            rospy.logerr(f"[step_policy] policy.getAction failed: {e}")
            return

        # Print action
        print("[step_policy] Next action:", action)

    def run(self):
        try:
            while not rospy.is_shutdown():
                # Non-blocking check for 's' key
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    c = sys.stdin.read(1)
                    if c.lower() == 's':
                        self.step()
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_term)
            rospy.loginfo("[step_policy] Terminal restored.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step through policy one action at a time')
    parser.add_argument('model_path', help='Path to the trained .pkl model file')
    args = parser.parse_args()

    rospy.init_node('step_policy', anonymous=False)
    executor = StepPolicy(args.model_path)
    executor.run()
