from image_segmentation.image_processor import ImageProcessor
from image_segmentation.clustering import ColorQuantizer
from image_segmentation.visualizer import ImageVisualizer
import os


def main():
    # Configuration parameters
    image_path = "data/images/test/3096.jpg"
    n_colors = 5  # Number of colors in the quantized output
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Specify the number of cores to use

    try:
        # Step 1: Load and preprocess the image
        # Convert from BGR to RGB and prepare for clustering
        processor = ImageProcessor()
        original_image = processor.load_image(image_path)
        normalized_image, image_1d = processor.preprocess_image(original_image)

        # Step 2: Perform color quantization
        # Reduce the number of colors using K-means clustering
        quantizer = ColorQuantizer(n_colors=n_colors)
        quantized_1d = quantizer.quantize_colors(image_1d)

        # Step 3: Reshape the result back to original dimensions
        quantized_image = quantized_1d.reshape(original_image.shape)

        # Step 4: Visualize results
        # Show original vs quantized images and the color palette
        visualizer = ImageVisualizer()
        visualizer.show_images(original_image, quantized_image, n_colors)
        visualizer.show_color_palette(quantizer.get_color_palette())

    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
