import torch
import cv2
import os
import numpy as np

video_folder = "main_data/loader_loading"
frame_folder = "frames/loader_loading"

# video_files = files = os.listdir(video_folder)
# print(files)

image_list = []

# Loop through each video file in the folder
for filename in os.listdir(video_folder):
    # Check if the file is a video
    if filename.endswith(".mp4"):
        # Full path to the video file
        filepath = os.path.join(video_folder, filename)

        # Open the video file
        cap = cv2.VideoCapture(filepath)

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Get the frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Wait time between frames in milliseconds
        wait_time = int(1000 / fps)

        # Loop through each frame in the video
        for i in range(total_frames):
            # Read the current frame
            ret, frame = cap.read()
            # Convert the colour images to greyscale in order to enable fast processing
            # frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            # Add some Gaussian Blur
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            # resize image
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            # Check if the frame was successfully read
            if ret:
                # image_list.append(frame)
                # Save the frame as an image file
                cv2.imwrite("{}/{}_{}.jpg".format(frame_folder, filename, i), frame)

                # Wait for the specified time before reading the next frame
                # cv2.waitKey(wait_time)