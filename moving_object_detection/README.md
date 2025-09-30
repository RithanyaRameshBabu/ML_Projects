# MovingObjectsDetector

A real-time motion detection system built with **Python**, **OpenCV**, and **imutils**.  
It detects movement through your laptop’s built-in camera (or an external webcam), highlights moving objects with bounding boxes, and displays detection status on screen.

---

## Features
- Capture video from your laptop camera or an external webcam
- Convert frames to grayscale and apply Gaussian blur for noise reduction
- Detect frame differences to identify motion
- Draw bounding boxes around moving objects
- Display detection status ("Normal" / "Moving Object detected")
- Real-time detection with live video feed

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RithanyaRameshBabu/ML_Projects.git
   cd ML_Projects/moving_object_detection

## Future Enhancements
- Save detected motion frames or recordings automatically
-Add sensitivity settings for different environments
-Integrate AI-based object recognition
-Send alerts on motion detection
-Add voice or sound alerts
-Detect motion from multiple cameras simultaneously
