import os
import cv2
import json
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
#from utils.removebg import segment_car
from utils.blur_predict import predict


@dataclass
class Frame:
    image: np.ndarray
    index: int
    blur_score: Optional[float] = None
    brightness_score: Optional[float] = None
    contrast_score: Optional[float] = None
    mask: Optional[np.ndarray] = None


class FrameExtractor:
    def __init__(self):
        self.blur_threshold = 0.5

    def extract_frames(self, video_path, num_frames, scene):
        video_cap = cv2.VideoCapture(video_path)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

        frames = self._get_frames(video_cap, frame_indices)
        self._assess_quality(frames)

        non_blur_frames = self._replace_blur_frames(video_cap, frames)
        self._save_frames(non_blur_frames, scene)

    def _get_frames(self, video_cap, frame_indices):
        frames = []
        for idx in frame_indices:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, image = video_cap.read()
            if success:
                frames.append(Frame(image=image, index=idx))
        return frames

    def _assess_quality(self, frames):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(lambda f: predict(img_array=f.image), frames)
            for frame, result in zip(frames, results):
                frame.blur_score = result[1]
                frame.brightness_score = result[2]
                frame.contrast_score = result[3]

    def _replace_blur_frames(self, video_cap, frames):
        new_frames = []
        for frame in frames:
            if frame.blur_score > self.blur_threshold:
                idx = min(frame.index + 5, int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, image = video_cap.read()
                if success:
                    new_frame = Frame(image=image, index=idx)
                    quality = predict(img_array=new_frame.image)
                    new_frame.blur_score, new_frame.brightness_score, new_frame.contrast_score = quality[1:4]
                    new_frames.append(new_frame)
                else:
                    new_frames.append(frame)
            else:
                new_frames.append(frame)
        return new_frames

    def _save_frames(self, frames, scene):
        images_path = os.path.join(scene, "images")
        masks_path = os.path.join(scene, "car_masks")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(masks_path, exist_ok=True)

        data = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, frame in enumerate(frames):
                mask = np.zeros(frame.image.shape[:2], dtype=np.uint8) #instead of passing black mask, you have to train a segmentation model to get masks
                image_file = f"{i:04d}.png"
                mask_file = f"{i:04d}.png"

                executor.submit(cv2.imwrite, os.path.join(images_path, image_file), frame.image)
                executor.submit(cv2.imwrite, os.path.join(masks_path, mask_file), mask)

                data[image_file] = {
                    "image_path": os.path.join(images_path, image_file),
                    "mask_path": os.path.join(masks_path, mask_file),
                    "blur_score": frame.blur_score,
                    "brightness_score": frame.brightness_score,
                    "contrast_score": frame.contrast_score
                }

        with open(os.path.join(scene, "frame_data.json"), "w") as f:
            json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video and save masks.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--num_frames", type=int, required=True, help="Number of frames to extract")
    parser.add_argument("--scene", type=str, required=True, help="Directory to save extracted frames and masks")

    args = parser.parse_args()

    extractor = FrameExtractor()
    extractor.extract_frames(args.video_path, args.num_frames, args.scene)


if __name__ == "__main__":
    main()