import cv2
import numpy as np


class ImageProcessor:
    """
    A class for handling image processing operations.
    Provides functionality for loading, color conversion, and normalization of images.
    This serves as the first step in our color quantization pipeline.
    """

    @staticmethod
    def load_image(image_path):
        """
        Loads an image and converts it from BGR to RGB color space.
        OpenCV loads images in BGR format by default, but we need RGB for
        better visualization and compatibility with other tools.

        Args:
            image_path (str): Path to the image file

        Returns:
            numpy.ndarray: Image in RGB format

        Raises:
            FileNotFoundError: If the specified image file cannot be found
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not find image at {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def preprocess_image(image):
        """
        Prepares the image for clustering by normalizing and reshaping.

        The preprocessing steps include:
        1. Normalizing pixel values to range [0,1] for better clustering
        2. Reshaping the 2D image array to 1D for K-means input

        Args:
            image (numpy.ndarray): Input image

        Returns:
            tuple: (normalized_image, reshaped_image)
                  - normalized_image: Image with pixel values in range [0,1]
                  - reshaped_image: Flattened array of pixels for clustering
        """
        # Normalize pixel values to [0,1] range for better numerical stability
        normalized = image / 255.0
        # Reshape image from (height, width, channels) to (pixels, channels)
        reshaped = normalized.reshape(-1, 3)
        return normalized, reshaped
