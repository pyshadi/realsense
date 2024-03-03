import os
import cv2
from object_detector import ObjectDetector
from realsense_camera import RealSenseCamera  

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    model_path = "../../../ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
    object_detector = ObjectDetector(model_path)
    camera = RealSenseCamera()

    try:
        while True:
            color_image = camera.get_frames()
            output_image = object_detector.detect(color_image)
            cv2.imshow('RealSense', output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        camera.stop()

if __name__ == '__main__':
    main()
