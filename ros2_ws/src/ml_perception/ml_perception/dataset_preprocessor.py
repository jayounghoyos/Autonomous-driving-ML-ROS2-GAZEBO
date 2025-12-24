#!/usr/bin/env python3
"""
Dataset Preprocessor for Autonomous Driving
Converts various datasets to unified format for training
"""

import os
import csv
import json
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class UdacityDatasetPreprocessor:
    """Process Udacity Self-Driving Car Dataset"""

    def __init__(self, dataset_path, output_path):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def process(self, target_size=(256, 256), use_all_cameras=True):
        """
        Process Udacity dataset

        Args:
            target_size: Resize images to this size
            use_all_cameras: Use left/right cameras with steering correction
        """
        csv_file = self.dataset_path / 'driving_log.csv'

        if not csv_file.exists():
            print(f"Error: {csv_file} not found")
            return

        df = pd.read_csv(csv_file)
        print(f"Total samples in dataset: {len(df)}")

        # Create output directories
        images_dir = self.output_path / 'images'
        images_dir.mkdir(exist_ok=True)

        processed_data = []
        sample_id = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            # Center camera
            center_path = self.dataset_path / row['center'].strip()
            if center_path.exists():
                img = cv2.imread(str(center_path))
                if img is not None:
                    # Resize
                    img_resized = cv2.resize(img, target_size)

                    # Save
                    img_filename = f"{sample_id:06d}.jpg"
                    cv2.imwrite(str(images_dir / img_filename), img_resized)

                    # Record data
                    processed_data.append({
                        'image': img_filename,
                        'steering': float(row['steering']),
                        'throttle': float(row['throttle']),
                        'brake': float(row['brake']),
                        'speed': float(row['speed'])
                    })
                    sample_id += 1

            if use_all_cameras:
                # Left camera (add steering correction)
                left_path = self.dataset_path / row['left'].strip()
                if left_path.exists():
                    img = cv2.imread(str(left_path))
                    if img is not None:
                        img_resized = cv2.resize(img, target_size)
                        img_filename = f"{sample_id:06d}.jpg"
                        cv2.imwrite(str(images_dir / img_filename), img_resized)

                        processed_data.append({
                            'image': img_filename,
                            'steering': float(row['steering']) + 0.25,  # Correction
                            'throttle': float(row['throttle']),
                            'brake': float(row['brake']),
                            'speed': float(row['speed'])
                        })
                        sample_id += 1

                # Right camera (subtract steering correction)
                right_path = self.dataset_path / row['right'].strip()
                if right_path.exists():
                    img = cv2.imread(str(right_path))
                    if img is not None:
                        img_resized = cv2.resize(img, target_size)
                        img_filename = f"{sample_id:06d}.jpg"
                        cv2.imwrite(str(images_dir / img_filename), img_resized)

                        processed_data.append({
                            'image': img_filename,
                            'steering': float(row['steering']) - 0.25,  # Correction
                            'throttle': float(row['throttle']),
                            'brake': float(row['brake']),
                            'speed': float(row['speed'])
                        })
                        sample_id += 1

        # Save metadata
        metadata_file = self.output_path / 'labels.csv'
        df_processed = pd.DataFrame(processed_data)
        df_processed.to_csv(metadata_file, index=False)

        print(f"\nProcessing complete!")
        print(f"Total processed samples: {len(processed_data)}")
        print(f"Images saved to: {images_dir}")
        print(f"Labels saved to: {metadata_file}")

        # Print statistics
        print(f"\nDataset statistics:")
        print(f"  Steering range: [{df_processed['steering'].min():.3f}, {df_processed['steering'].max():.3f}]")
        print(f"  Mean steering: {df_processed['steering'].mean():.3f}")
        print(f"  Speed range: [{df_processed['speed'].min():.1f}, {df_processed['speed'].max():.1f}]")


class DataAugmentation:
    """Data augmentation for driving images"""

    @staticmethod
    def random_brightness(image, factor=0.3):
        """Randomly adjust brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ratio = 1.0 + np.random.uniform(-factor, factor)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def random_shadow(image):
        """Add random shadow"""
        h, w = image.shape[:2]
        x1, y1 = w * np.random.rand(), 0
        x2, y2 = w * np.random.rand(), h

        xm, ym = np.mgrid[0:h, 0:w]
        mask = np.zeros_like(image[:, :, 0])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio

        return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

    @staticmethod
    def random_flip(image, steering):
        """Randomly flip image horizontally"""
        if np.random.rand() < 0.5:
            return cv2.flip(image, 1), -steering
        return image, steering

    @staticmethod
    def random_shift(image, steering, shift_range=100, steering_per_pixel=0.002):
        """Randomly shift image"""
        rows, cols, _ = image.shape
        tx = shift_range * np.random.uniform(-0.5, 0.5)
        ty = shift_range * np.random.uniform(-0.5, 0.5)

        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (cols, rows))

        steering_adjust = steering + steering_per_pixel * tx
        return image, steering_adjust


def main():
    """Main preprocessing function"""
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess driving datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset (Udacity format)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--size', type=int, nargs=2, default=[256, 256],
                        help='Target image size (width height)')
    parser.add_argument('--all-cameras', action='store_true',
                        help='Use all cameras (left, center, right)')

    args = parser.parse_args()

    preprocessor = UdacityDatasetPreprocessor(args.dataset, args.output)
    preprocessor.process(
        target_size=tuple(args.size),
        use_all_cameras=args.all_cameras
    )


if __name__ == '__main__':
    main()
