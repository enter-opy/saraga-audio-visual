import cv2
import numpy as np
from IPython.display import Video, display, HTML
# Load keypoints and scores
import os

def process_frames(keypoints_file, scores_file, video_file, temp_output_file):
    keypoints = np.load(keypoints_file)
    scores = np.load(scores_file)

    # Skeleton for 135 keypoints (MMPose)
    skeleton = [
        (0, 1), (1, 2),  # Eyes (left to right)
        (0, 3), (0, 4),  # Nose to ears (left and right)
        (5, 6),          # Shoulders (left and right)
        (5, 7), (7, 9),  # Left arm (shoulder -> elbow -> wrist)
        (6, 8), (8, 10),
        (11,12), # Right arm (shoulder -> elbow -> wrist)
        (5, 11), (6, 12), # Shoulders to hips
        (11, 13), (13, 15), # Left leg (hip -> knee -> ankle)
        (12, 14), (14, 16)  # Right leg (hip -> knee -> ankle)
    ]

    # Open video file
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define start and end frames for the 20-second segment
    start_time = 10  # Start time in seconds (adjust as needed)
    end_time = start_time + 20  # End time in seconds
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Temporary output video file
    out = cv2.VideoWriter(temp_output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process the selected frames
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= frame_idx < end_frame:
            # Get keypoints and scores for the current frame
            if frame_idx < len(keypoints):
                frame_keypoints = keypoints[frame_idx]
                frame_scores = scores[frame_idx]
                # Draw keypoints and skeleton
                for i, (x, y) in enumerate(frame_keypoints):
                    # Only draw if confidence score is above threshold
                    if frame_scores[i] > 0.5:  # Adjust threshold as needed
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                # Draw skeleton
                for connection in skeleton:
                    start, end = connection
                    if frame_scores[start] > 0.5 and frame_scores[end] > 0.5:
                        x1, y1 = frame_keypoints[start]
                        x2, y2 = frame_keypoints[end]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # Write frame to output video
            out.write(frame)
        frame_idx += 1
        # Stop processing after the end frame
        if frame_idx >= end_frame:
            break

    cap.release()
    cv2.destroyAllWindows()
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
