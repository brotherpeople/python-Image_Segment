from sklearn.cluster import KMeans
import numpy as np


class ColorQuantizer:
    """
    Implements color quantization using K-means clustering.
    Reduces the number of unique colors in an image while maintaining
    visual similarity to the original image.
    """

    def __init__(self, n_colors=5):
        """
        Initialize the quantizer with the desired number of colors.

        Args:
            n_colors (int): Number of colors in the quantized output.
                           Higher values preserve more detail but increase complexity.
        """
        self.n_colors = n_colors
        self.model = None

    def quantize_colors(self, image_1d):
        """
        Quantizes the image colors using K-means clustering.
        Each pixel color is replaced with its nearest cluster center.

        Args:
            image_1d (numpy.ndarray): Flattened image array of shape (pixels, 3)

        Returns:
            numpy.ndarray: Quantized image data where each pixel is mapped
                          to its nearest cluster center
        """
        self.model = KMeans(n_clusters=self.n_colors, random_state=42)
        self.model.fit(image_1d)

        # Map each pixel to its closest cluster center
        quantized = self.model.cluster_centers_[self.model.labels_]
        return quantized

    def get_color_palette(self):
        """
        Retrieves the representative colors found by K-means clustering.
        These colors form the palette used in the quantized image.

        Returns:
            numpy.ndarray: Array of RGB values for each cluster center

        Raises:
            ValueError: If called before performing quantization
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        return self.model.cluster_centers_
