#!/usr/bin/env python3
"""
Waymo Open Dataset E2E TFRecord Parser
Extracts camera images and driving commands from Waymo E2E dataset
"""

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm


class WaymoE2EParser:
    """Parse Waymo End-to-End Driving TFRecords"""

    def __init__(self, tfrecord_path):
        self.tfrecord_path = Path(tfrecord_path)

        # Feature description for Waymo E2E dataset
        # Based on: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_vision_based_e2e_driving.ipynb
        self.feature_description = {
            # Camera images (JPEG encoded)
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),

            # Driving command (high-level action)
            'command': tf.io.FixedLenFeature([], tf.int64),

            # Vehicle state
            'vehicle/speed': tf.io.FixedLenFeature([], tf.float32),
            'vehicle/steering_angle': tf.io.FixedLenFeature([], tf.float32),

            # Scenario metadata
            'scenario_id': tf.io.FixedLenFeature([], tf.string),
            'timestamp_micros': tf.io.FixedLenFeature([], tf.int64),
        }

    def parse_example(self, example_proto):
        """Parse a single example from TFRecord"""
        return tf.io.parse_single_example(example_proto, self.feature_description)

    def extract_to_dataset(self, output_dir, max_examples=None):
        """
        Extract TFRecord to images + CSV format

        Args:
            output_dir: Output directory path
            max_examples: Max number of examples to extract (None = all)
        """
        output_path = Path(output_dir)
        images_dir = output_path / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        dataset = tf.data.TFRecordDataset(str(self.tfrecord_path))

        extracted_data = []

        print(f"Extracting from {self.tfrecord_path.name}...")

        for idx, raw_record in enumerate(tqdm(dataset)):
            if max_examples and idx >= max_examples:
                break

            try:
                example = self.parse_example(raw_record)

                # Decode JPEG image
                image_encoded = example['image/encoded'].numpy()
                image = cv2.imdecode(
                    np.frombuffer(image_encoded, np.uint8),
                    cv2.IMREAD_COLOR
                )

                if image is None:
                    print(f"Warning: Failed to decode image at index {idx}")
                    continue

                # Save image
                image_filename = f"{idx:06d}.jpg"
                cv2.imwrite(str(images_dir / image_filename), image)

                # Extract metadata
                # scenario_id is binary data, convert to hex string
                scenario_bytes = example['scenario_id'].numpy()
                scenario_id_str = scenario_bytes.hex() if isinstance(scenario_bytes, bytes) else str(scenario_bytes)

                data_entry = {
                    'image': image_filename,
                    'steering_angle': float(example['vehicle/steering_angle'].numpy()),
                    'speed': float(example['vehicle/speed'].numpy()),
                    'command': int(example['command'].numpy()),
                    'scenario_id': scenario_id_str,
                    'timestamp': int(example['timestamp_micros'].numpy()),
                    'height': int(example['image/height'].numpy()),
                    'width': int(example['image/width'].numpy()),
                }

                extracted_data.append(data_entry)

            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                continue

        # Save metadata as JSON
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(extracted_data, f, indent=2)

        # Also save as CSV for easier loading
        import pandas as pd
        df = pd.DataFrame(extracted_data)
        df.to_csv(output_path / 'labels.csv', index=False)

        print(f"\nExtraction complete!")
        print(f"  Total examples: {len(extracted_data)}")
        print(f"  Images saved to: {images_dir}")
        print(f"  Labels saved to: {output_path / 'labels.csv'}")
        print(f"  Metadata saved to: {metadata_file}")

        # Print statistics
        if len(extracted_data) > 0:
            print(f"\nDataset statistics:")
            print(f"  Steering angle range: [{df['steering_angle'].min():.3f}, {df['steering_angle'].max():.3f}]")
            print(f"  Mean steering: {df['steering_angle'].mean():.3f}")
            print(f"  Speed range: [{df['speed'].min():.1f}, {df['speed'].max():.1f}] m/s")
            print(f"  Commands: {df['command'].value_counts().to_dict()}")

        return extracted_data


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Extract Waymo E2E TFRecords')
    parser.add_argument('--tfrecord', type=str, required=True,
                        help='Path to TFRecord file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--max-examples', type=int, default=None,
                        help='Maximum number of examples to extract (default: all)')

    args = parser.parse_args()

    parser_obj = WaymoE2EParser(args.tfrecord)
    parser_obj.extract_to_dataset(args.output, max_examples=args.max_examples)


if __name__ == '__main__':
    main()
