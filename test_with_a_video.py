import cv2
import numpy as np
import supervision as sv
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Set up the configuration and load the model
cfg = get_cfg()
cfg.merge_from_file("CONFIG_PATH")  # Replace with the actual path to your config file
cfg.MODEL.WEIGHTS = "WEIGHTS_PATH"  # Replace with the actual path to your model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for inference
predictor = DefaultPredictor(cfg)

# Define your class names
CLASS_NAMES = ['car', 'truck', 'pedestrian', 'bicyclist', 'lights']

# Define the frame processing function
def process_frame(frame: np.ndarray, _) -> np.ndarray:
    # Run inference on the frame
    result = predictor(frame)
    
    # Get detections from Detectron2 result
    detections = sv.Detections.from_detectron2(result)
    
    # Annotate boxes and labels (kept thickness only)
    box_annotator = sv.BoxAnnotator(thickness=4)

    # Generate labels for each detection: Class name and confidence score
    labels = [
        f"{CLASS_NAMES[class_id]} {confidence:0.2f}" 
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    
    # Annotate the frame with bounding boxes and labels
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    
    return frame

# Process the video and save the output
VIDEO_PATH = "/content/drive/MyDrive/test_video.mp4"  # Path to your input video
sv.process_video(source_path=VIDEO_PATH, target_path="result.mp4", callback=process_frame)

print("Video processing complete! Output saved as result.mp4")
