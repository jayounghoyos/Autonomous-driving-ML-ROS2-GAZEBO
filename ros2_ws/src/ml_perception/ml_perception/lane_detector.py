#!/usr/bin/env python3
"""
Lane Detection Node for Autonomous Driving
Uses traditional computer vision (Canny + Hough Transform)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np


class LaneDetector(Node):
    """ROS 2 Node for lane detection using computer vision"""

    def __init__(self):
        super().__init__('lane_detector')

        # Parameters
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('visualization_topic', '/lane_visualization')
        self.declare_parameter('lane_center_topic', '/lane_center')
        self.declare_parameter('steering_angle_topic', '/lane_steering_angle')
        self.declare_parameter('enable_visualization', True)

        input_topic = self.get_parameter('input_topic').value
        self.viz_topic = self.get_parameter('visualization_topic').value
        lane_center_topic = self.get_parameter('lane_center_topic').value
        steering_angle_topic = self.get_parameter('steering_angle_topic').value
        self.enable_viz = self.get_parameter('enable_visualization').value

        # Lane detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 50
        self.hough_min_line_length = 40
        self.hough_max_line_gap = 100

        # ROS 2 subscribers and publishers
        self.image_subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )

        self.lane_center_publisher = self.create_publisher(
            Point,
            lane_center_topic,
            10
        )

        self.steering_angle_publisher = self.create_publisher(
            Float32,
            steering_angle_topic,
            10
        )

        if self.enable_viz:
            self.visualization_publisher = self.create_publisher(
                Image,
                self.viz_topic,
                10
            )

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Image dimensions (will be set from first frame)
        self.image_height = None
        self.image_width = None

        self.get_logger().info(
            f'Lane Detector initialized\n'
            f'  Input topic: {input_topic}\n'
            f'  Lane center topic: {lane_center_topic}\n'
            f'  Steering angle topic: {steering_angle_topic}\n'
            f'  Visualization enabled: {self.enable_viz}'
        )

    def region_of_interest(self, img, vertices):
        """Apply region of interest mask to image"""
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self, img, lines, color=(0, 255, 0), thickness=3):
        """Draw lines on image"""
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def calculate_lane_lines(self, lines):
        """Calculate left and right lane lines from detected segments"""
        if lines is None:
            return None, None

        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Skip horizontal lines
            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Filter based on slope
            if abs(slope) < 0.3:  # Too horizontal
                continue

            if slope < 0:  # Left lane
                left_lines.append((slope, intercept))
            else:  # Right lane
                right_lines.append((slope, intercept))

        # Average lines
        left_lane = np.mean(left_lines, axis=0) if left_lines else None
        right_lane = np.mean(right_lines, axis=0) if right_lines else None

        return left_lane, right_lane

    def make_lane_coordinates(self, line_params, y1, y2):
        """Convert line parameters to coordinates"""
        if line_params is None:
            return None

        slope, intercept = line_params
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def calculate_steering_angle(self, left_line, right_line):
        """Calculate steering angle from lane lines"""
        if left_line is None and right_line is None:
            return 0.0  # No lanes detected, go straight

        # Image center
        image_center = self.image_width / 2

        if left_line is not None and right_line is not None:
            # Both lanes detected - steer between them
            left_x = left_line[2]  # x at bottom of image
            right_x = right_line[2]
            lane_center = (left_x + right_x) / 2
        elif left_line is not None:
            # Only left lane - keep distance from it
            lane_center = left_line[2] + 100  # Offset to right
        else:
            # Only right lane - keep distance from it
            lane_center = right_line[2] - 100  # Offset to left

        # Calculate steering angle (positive = right, negative = left)
        offset = lane_center - image_center
        steering_angle = offset / (self.image_width / 2) * 0.5  # Normalize to [-0.5, 0.5]

        return float(steering_angle)

    def image_callback(self, msg: Image):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Set image dimensions from first frame
            if self.image_height is None:
                self.image_height, self.image_width = cv_image.shape[:2]

            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Canny edge detection
            edges = cv2.Canny(blur, self.canny_low, self.canny_high)

            # Define region of interest (trapezoid)
            height = self.image_height
            width = self.image_width
            roi_vertices = np.array([[
                (int(width * 0.1), height),
                (int(width * 0.4), int(height * 0.6)),
                (int(width * 0.6), int(height * 0.6)),
                (int(width * 0.9), height)
            ]], dtype=np.int32)

            # Apply ROI mask
            masked_edges = self.region_of_interest(edges, roi_vertices)

            # Hough line detection
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=2,
                theta=np.pi/180,
                threshold=self.hough_threshold,
                minLineLength=self.hough_min_line_length,
                maxLineGap=self.hough_max_line_gap
            )

            # Calculate left and right lanes
            left_lane, right_lane = self.calculate_lane_lines(lines)

            # Convert to coordinates
            y1 = self.image_height
            y2 = int(self.image_height * 0.6)
            left_line = self.make_lane_coordinates(left_lane, y1, y2)
            right_line = self.make_lane_coordinates(right_lane, y1, y2)

            # Calculate steering angle
            steering_angle = self.calculate_steering_angle(left_line, right_line)

            # Publish steering angle
            angle_msg = Float32()
            angle_msg.data = steering_angle
            self.steering_angle_publisher.publish(angle_msg)

            # Publish lane center
            if left_line is not None and right_line is not None:
                center_x = (left_line[2] + right_line[2]) / 2
                center_msg = Point()
                center_msg.x = float(center_x)
                center_msg.y = float(self.image_height)
                center_msg.z = 0.0
                self.lane_center_publisher.publish(center_msg)

            # Visualization
            if self.enable_viz:
                line_image = cv_image.copy()

                # Draw ROI
                cv2.polylines(line_image, roi_vertices, True, (255, 0, 0), 2)

                # Draw detected lanes
                if left_line is not None:
                    cv2.line(line_image, (left_line[0], left_line[1]),
                            (left_line[2], left_line[3]), (0, 255, 0), 5)
                if right_line is not None:
                    cv2.line(line_image, (right_line[0], right_line[1]),
                            (right_line[2], right_line[3]), (0, 255, 0), 5)

                # Draw lane center
                if left_line is not None and right_line is not None:
                    center_x = int((left_line[2] + right_line[2]) / 2)
                    cv2.circle(line_image, (center_x, self.image_height - 50),
                             10, (0, 0, 255), -1)
                    cv2.line(line_image, (int(width/2), self.image_height - 50),
                            (center_x, self.image_height - 50), (0, 0, 255), 3)

                # Add text info
                cv2.putText(line_image, f'Steering: {steering_angle:.3f}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2)

                # Convert and publish
                viz_msg = self.bridge.cv2_to_imgmsg(line_image, encoding='bgr8')
                viz_msg.header = msg.header
                self.visualization_publisher.publish(viz_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}',
                                   throttle_duration_sec=5)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
