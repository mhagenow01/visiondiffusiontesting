#!/usr/bin/env python3
import os
import sys
import threading
import select
import termios
import tty
import pickle

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DataLogger:
    def __init__(self, freq_hz=5.0):
        self.recording = False
        self.states = []    # list of np.array shape (7,)
        self.images = []    # list of np.array shape (H,W,3)
        self.current_pose = None
        self.current_image = None

        self.bridge = CvBridge()
        rospy.Subscriber('/ur5e/compliant_pose', PoseStamped, self._pose_cb)
        rospy.Subscriber('/robot_webcam/image_raw', Image,        self._img_cb)

        # timer for fixed-rate sampling
        self.timer = rospy.Timer(rospy.Duration(1.0 / freq_hz), self._timer_cb)

        # start key‐listener thread
        t = threading.Thread(target=self._key_listener)
        t.daemon = True
        t.start()

        rospy.loginfo(f"[data_logger] Ready. Press 'r' to start/stop recording at {freq_hz} Hz.")

    def _pose_cb(self, msg: PoseStamped):
        # stash the latest pose
        self.current_pose = msg

    def _img_cb(self, msg: Image):
        # stash the latest image
        self.current_image = msg

    def _timer_cb(self, event):
        if not self.recording:
            return
        if (self.current_pose is None) or (self.current_image is None):
            return

        # convert pose → 7-d vector
        p = self.current_pose.pose
        vec = np.array([
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
        ], dtype=np.float32)
        self.states.append(vec)

        # convert ROS Image → RGB numpy
        cv_img = self.bridge.imgmsg_to_cv2(self.current_image, desired_encoding='rgb8')
        self.images.append(cv_img)

    def _key_listener(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            while not rospy.is_shutdown():
                # non-blocking check for keypress
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    c = sys.stdin.read(1)
                    if c.lower() == 'r':
                        self._toggle_recording()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _toggle_recording(self):
        self.recording = not self.recording
        state = "STARTED" if self.recording else "STOPPED"
        rospy.loginfo(f"[data_logger] Recording {state}")
        if not self.recording:
            self._save_episode()

    def _save_episode(self):
        N = len(self.states)
        if N < 2:
            rospy.logwarn("[data_logger] Too few samples, skipping save")
            self.states.clear()
            self.images.clear()
            return

        # align: state[t] → action[t] = state[t+1]
        states_arr  = np.stack(self.states[:-1], axis=1)  # (7, N−1)
        actions_arr = np.stack(self.states[1: ], axis=1)  # (7, N−1)
        imgs_arr    = np.stack(self.images[:-1], axis=0)  # (N−1, H, W, 3)

        timestamp = rospy.Time.now().to_nsec()
        fname = f"episode_{timestamp}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump((states_arr, actions_arr, imgs_arr), f)
        rospy.loginfo(f"[data_logger] Saved episode: {fname}")

        # reset buffers
        self.states.clear()
        self.images.clear()

if __name__ == '__main__':
    rospy.init_node('data_logger', anonymous=False)
    freq = rospy.get_param('~freq', 5.0)
    logger = DataLogger(freq_hz=freq)
    rospy.spin()