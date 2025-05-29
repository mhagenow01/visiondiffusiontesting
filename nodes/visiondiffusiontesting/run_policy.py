#!/usr/bin/env python
import os
import argparse
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# adjust this import to wherever your Diffusion class lives
# from policies.diffusion_with_images import Diffusion, normalize_data
from vision_diffusion import DiffusionVision

class PolicyExecutor:
    def __init__(self, model_path, freq_hz):
        # 1) load policy
        self.policy = Diffusion()  
        self.policy.load(model_path)
        rospy.loginfo(f"[executor] Loaded diffusion policy from {model_path}")

        # 2) state holders
        self.current_pose   = None  # PoseStamped
        self.current_image  = None  # sensor_msgs/Image
        self.bridge = CvBridge()

        # 3) ROS comms
        rospy.Subscriber('/ur5e/compliant_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/robot_webcam/image_raw',  Image,       self.image_cb)
        self.cmd_pub = rospy.Publisher('/ur5e/command_pose', PoseStamped, queue_size=1)

        # 4) timer for inference at fixed rate
        self.timer = rospy.Timer(rospy.Duration(1.0/freq_hz), self.timer_cb)
        rospy.loginfo(f"[executor] Will run policy at {freq_hz} Hz")

    def pose_cb(self, msg: PoseStamped):
        self.current_pose = msg

    def image_cb(self, msg: Image):
        self.current_image = msg

    def timer_cb(self, event):
        # only run when we have both a pose and an image
        if self.current_pose is None or self.current_image is None:
            return

        # 1) extract state vector (7-dim: x,y,z,qx,qy,qz,qw)
        p = self.current_pose.pose
        state_vec = np.array([
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
        ], dtype=np.float32)

        # 2) convert ROS image → RGB numpy array
        img_np = self.bridge.imgmsg_to_cv2(self.current_image, desired_encoding='rgb8')

        # 3) get next action (pose) from policy
        try:
            action = self.policy.getAction(state_vec, img_np, forecast=False)
        except Exception as e:
            rospy.logerr(f"[executor] policy.getAction failed: {e}")
            return

        # 4) build PoseStamped and publish
        cmd = PoseStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = self.current_pose.header.frame_id  # keep the same frame
        cmd.pose.position.x = float(action[0])
        cmd.pose.position.y = float(action[1])
        cmd.pose.position.z = float(action[2])
        cmd.pose.orientation.x = float(action[3])
        cmd.pose.orientation.y = float(action[4])
        cmd.pose.orientation.z = float(action[5])
        cmd.pose.orientation.w = float(action[6])

        self.cmd_pub.publish(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to your trained .pkl model")
    parser.add_argument("--freq", type=float, default=5.0,
                        help="inference rate in Hz (default: 5.0)")
    args = parser.parse_args()

    rospy.init_node('policy_executor', anonymous=False)
    executor = PolicyExecutor(model_path=args.model_path, freq_hz=args.freq)
    rospy.spin()

