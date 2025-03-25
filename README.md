# GS_videoto3Dpipeline
##  Requirements

- Python 3.8+
- OpenCV
- NumPy
- Threading/Concurrent Futures (standard)
- A `predict()` function in `utils/blur_predict.py` for quality scoring

Install dependencies:
```bash
pip install -r requriment.txt


python video2frame.py \
  --video_path path/to/video.mp4 \
  --num_frames 360 \
  --scene output/scene_name