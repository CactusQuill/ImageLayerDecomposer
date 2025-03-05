# ColorSep: Textile Color Separation Tool

ColorSep is an advanced color separation tool specifically designed for textile printing applications. It extracts different color layers from an image, allowing printers to create separate screens for each color in the printing process.

## Features

- **Multiple Color Separation Methods**:
  - **K-means clustering**: Segments the image into distinct color clusters
  - **Dominant color extraction**: Extracts the most common colors from the image
  - **Color thresholding**: Uses thresholds to separate colors
  - **LAB color space**: Uses perceptual color differences for more accurate separation

- **Advanced Image Processing**:
  - Noise reduction
  - Edge-preserving smoothing
  - Sharpening
  - Connected component analysis

- **Customizable Options**:
  - Adjust number of colors to extract
  - Control color sensitivity and compactness
  - Set background color
  - Fine-tune post-processing parameters

- **Intuitive UI**:
  - Real-time preview of color layers
  - Color information display (RGB, Hex, coverage percentage)
  - Combined preview of all layers
  - Downloadable results as individual PNGs or ZIP package

- **Advanced Layer Manipulation**:
  - Combine two layers into a new one
  - Change colors of layers with professional color codes
  - Support for Pantone TPX and TPG color systems
  - Exact color extraction for pixel-perfect separation
  - Control layer order and visibility for complex designs
  - Preview different layer combinations with toggling options

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Upload an image using the sidebar file uploader
3. Choose a color separation method and adjust parameters
4. View the extracted color layers
5. Use the Layer Order & Visibility Settings to:
   - Change the stacking order of layers
   - Toggle layer visibility on/off
   - Preview different layer combinations
6. Download individual layers or all layers as a zip file with order preserved

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- scikit-learn
- scikit-image
- Matplotlib
- Pillow

## How It Works

The application uses several advanced image processing techniques to separate colors:

1. **Exact Color Extraction**: Creates one layer per unique color for pixel-perfect separation
2. **K-means Clustering**: Uses unsupervised learning to group similar colors together, creating distinct clusters
3. **LAB Color Space**: Works in a perceptually uniform color space to better match human color perception
4. **Connected Component Analysis**: Identifies and extracts continuous regions of similar colors
5. **Morphological Operations**: Removes noise and refines color regions
6. **Professional Color Systems**: Supports Pantone TPX and TPG color codes used in the textile industry

## Applications in Textile Printing

This tool is particularly useful for textile printing processes that require separate screens or plates for each color, including:

- Screen printing
- Digital textile printing
- Block printing
- Transfer printing

By separating an image into color-specific layers, printers can create individual screens or plates for each color, resulting in more precise and efficient printing processes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The scikit-learn team for their implementation of K-means clustering
- The OpenCV team for their comprehensive computer vision library
- The Streamlit team for their easy-to-use web application framework
