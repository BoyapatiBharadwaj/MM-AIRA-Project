import cv2
from PIL import Image
import numpy as np

class CVProcessor:
    def __init__(self):
        pass

    def read_image(self, file_path):
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    def simple_object_detection(self, image):
        # Example: just convert to grayscale and detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges
