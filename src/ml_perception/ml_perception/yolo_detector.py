#!/usr/bin/env python3
"""
YOLO Object Detection Node for Autonomous Driving
Detects obstacles using YOLOv8 with ONNX Runtime GPU acceleration
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time


class YOLODetector(Node):
    """ROS 2 Node for YOLO-based object detection"""

    def __init__(self):
        super().__init__('yolo_detector')

        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')  # nano model for speed
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/detections')
        self.declare_parameter('visualization_topic', '/yolo_visualization')
        self.declare_parameter('enable_visualization', True)

        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.viz_topic = self.get_parameter('visualization_topic').value
        self.enable_viz = self.get_parameter('enable_visualization').value

        # Initialize YOLO model
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        try:
            self.model = YOLO(model_path)
            self.get_logger().info('YOLO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            raise

        # ROS 2 subscribers and publishers
        self.image_subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )

        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            output_topic,
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

        # Performance tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0

        self.get_logger().info(
            f'YOLO Detector initialized\n'
            f'  Input topic: {input_topic}\n'
            f'  Output topic: {output_topic}\n'
            f'  Confidence threshold: {self.confidence_threshold}\n'
            f'  Visualization enabled: {self.enable_viz}'
        )

    def image_callback(self, msg: Image):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run YOLO inference
            results = self.model(cv_image, conf=self.confidence_threshold, verbose=False)

            # Create detection message
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            # Process detections
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i, box in enumerate(boxes):
                    detection = Detection2D()
                    detection.header = msg.header

                    # Bounding box (xmin, ymin, xmax, ymax)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detection.bbox.center.x = float((x1 + x2) / 2)
                    detection.bbox.center.y = float((y1 + y2) / 2)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)

                    # Class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = self.model.names[class_id]

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(class_id)
                    hypothesis.hypothesis.score = confidence
                    detection.results.append(hypothesis)

                    detections_msg.detections.append(detection)

                    self.get_logger().debug(
                        f'Detected: {class_name} (ID: {class_id}) '
                        f'- Confidence: {confidence:.2f}'
                    )

            # Publish detections
            self.detection_publisher.publish(detections_msg)

            # Publish visualization if enabled
            if self.enable_viz:
                annotated_frame = results[0].plot()

                # Add FPS counter
                self.frame_count += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = time.time()

                cv2.putText(
                    annotated_frame,
                    f'FPS: {self.fps:.1f}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

                # Convert back to ROS message
                viz_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                viz_msg.header = msg.header
                self.visualization_publisher.publish(viz_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}', throttle_duration_sec=5)


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
