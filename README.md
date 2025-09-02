# Image Color Quantization Project
&nbsp;

<div align="center">
  <img width="600" height="215" alt="image" src="https://github.com/user-attachments/assets/d3049ce6-dea8-40cb-be09-30e425aa0398" />
  <img width="600" height="250" alt="image" src="https://github.com/user-attachments/assets/f491c2dd-3750-4c18-9271-558adab60960" />
</div>
&nbsp;

## Overview

This project implements color quantization using K-means clustering to reduce the number of colors in images while maintaining visual similarity. Created as part of computer science undergraduate studies to understand image processing and clustering algorithms.

## Features

-   Color space conversion (BGR to RGB)
-   Image normalization and preprocessing
-   K-means clustering for color reduction
-   Interactive visualization of results
-   Extracted color palette display

## Project Structure

```
python-Image_Segment/
├── image_segmentation/
│   ├── image_processor.py    # Image loading and preprocessing
│   ├── clustering.py         # K-means color quantization
│   └── visualizer.py         # Result visualization
├── data/
│   └── images/              # Test images
│       |── test/
│       └── train/
├── ImageSegment.py          # Initial single script of the project
├── main.py                  # Main execution script
└── requirements.txt         # Dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/brotherpeople/python-Image_Segment.git
cd python-Image_Segment

# Install dependencies
pip install numpy opencv-python scikit-learn matplotlib
```

## Usage

```python
python main.py
```

The script will:

1. Load and preprocess the input image
2. Apply K-means clustering to reduce colors
3. Display original and quantized images
4. Show the extracted color palette

## Results

-   Reduces image colors while preserving visual quality
-   Extracts dominant colors from complex images
-   Provides visual comparison between original and processed images

## Technical Implementation

-   Image processing using OpenCV
-   Color space handling with NumPy
-   K-means clustering from scikit-learn
-   Visualization with Matplotlib

## Development Process

This project evolved from a single script (`ImageSegment.py`) into a well-structured Python package. The development process was assisted by Claude AI, which helped with code organization, documentation, and best practices implementation.

## Future Improvements

-   Add support for batch processing
-   Implement additional clustering algorithms
-   Create GUI interface
-   Add image export functionality

## License

MIT License

## Acknowledgments

-   Base implementation started from `ImageSegment.py`
-   Project structure and documentation enhanced with Claude AI assistance
-   Test images from Berkeley Segmentation Dataset

This project demonstrates practical applications of clustering algorithms in image processing while maintaining code organization and documentation standards.
