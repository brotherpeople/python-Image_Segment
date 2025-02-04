import matplotlib.pyplot as plt
import numpy as np


class ImageVisualizer:
    """
    Handles the visualization of original and quantized images,
    as well as the extracted color palette.
    Provides methods to create clear, informative visual comparisons.
    """

    @staticmethod
    def show_images(original, quantized, n_colors):
        """
        Displays the original and quantized images side by side.
        This allows for easy visual comparison of the quantization effects.

        Args:
            original (numpy.ndarray): Original RGB image
            quantized (numpy.ndarray): Quantized RGB image
            n_colors (int): Number of colors used in quantization
        """
        plt.figure(figsize=(10, 5))

        # Display original image on the left
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.axis("off")
        plt.imshow(original)

        # Display quantized image on the right
        plt.subplot(1, 2, 2)
        plt.title(f"Quantized Image ({n_colors} colors)")
        plt.axis("off")
        plt.imshow(quantized)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_color_palette(colors):
        """
        Visualizes the extracted color palette.
        Shows the representative colors found by the clustering algorithm.

        Args:
            colors (numpy.ndarray): Array of RGB color values representing
                                  the extracted color palette
        """
        plt.figure(figsize=(8, 2))
        plt.title("Extracted Color Palette")

        # Display each color in the palette
        for i, color in enumerate(colors):
            plt.subplot(1, len(colors), i + 1)
            plt.axis("off")
            plt.imshow([[color]])

        plt.tight_layout()
        plt.show()
