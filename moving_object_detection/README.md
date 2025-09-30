# MovingObjectsDetector

A simple real-time motion detection system built with **Python**, **OpenCV**, and **imutils**.  
It detects movement through webcam video feed, highlights moving objects with bounding boxes, and displays detection status on screen.

## Features
- Capture video from webcam
- Convert frames to grayscale and apply Gaussian blur
- Detect frame differences to identify motion
- Draw bounding boxes around moving objects
- Display detection status ("Normal" / "Moving Object detected")