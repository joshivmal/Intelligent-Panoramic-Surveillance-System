from __future__ import print_function
from pyimagesearch.basicmotiondetector import BasicMotionDetector
from pyimagesearch.panorama import Stitcher
import numpy as np
import imutils
import cv2


# Paths to recorded videos
left_video_path = "Videos/left.mp4"  # Update with your file path
right_video_path = "Videos/right.mp4"  # Update with your file path

# Open video files
print("[INFO] Opening video files...")
leftStream = cv2.VideoCapture(left_video_path)
rightStream = cv2.VideoCapture(right_video_path)

# Initialize the image stitcher and motion detector
stitcher = Stitcher()
motion = BasicMotionDetector(minArea=15000)

while True:
    # Read frames from both videos
    ret1, left = leftStream.read()
    ret2, right = rightStream.read()
    
    # Check if frames are read successfully
    if not ret1 or not ret2:
        print("[INFO] End of one or both videos.")
        break

    # Resize frames for processing
    left = imutils.resize(left, width=400)
    right = imutils.resize(right, width=400)

    # Stitch frames together
    result = stitcher.stitch([left, right])

    if result is None:
        print("[INFO] Homography could not be computed")
        continue

    # Display output frames
    cv2.imshow("Stitched Result", result)
    cv2.imshow("Left Video", left)
    cv2.imshow("Right Video", right)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
print("[INFO] Cleaning up...")
cv2.destroyAllWindows()
leftStream.release()
rightStream.release()


# print(cv2.__version__) 
