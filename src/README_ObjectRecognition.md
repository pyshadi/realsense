# Object Detection and Recognition using RealSense camera
This is a Python script for object detection and recognition using the RealSense camera and TensorFlow Object Detection API. <br>
It detects various objects in the image and displays the label of the detected objects on the video stream.

## Dependences
* Python 3.x
* TensorFlow
* OpenCV
* PyRealSense2

## Usage
* Clone or download the repository.
* Run the script using the command: python object_detection.py <br>
```
        image = camera.read()
        output_image = object_detector.detect(image)
```