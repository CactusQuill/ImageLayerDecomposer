import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
from skimage import color
from sklearn.cluster import KMeans
import tempfile
import os
import zipfile
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="ColorSep - Textile Color Separation Tool",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 10px;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
    }
    .color-chip {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 5px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>ColorSep: Textile Color Separation Tool</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Upload an image and extract different color layers for textile printing</p>", unsafe_allow_html=True)

# Import color separation functions from external modules
from color_separation import (
    kmeans_color_separation,
    dominant_color_separation, 
    threshold_color_separation,
    lab_color_separation
)

# Sidebar for controls
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        method = st.selectbox(
            "Choose separation method",
            [
                "K-means clustering", 
                "Dominant color extraction",
                "Color thresholding",
                "LAB color space"
            ]
        )
        
        # Parameters for K-means
        if method == "K-means clustering":
            num_colors = st.slider("Number of colors to extract", 2, 20, 5)
            compactness = st.slider("Color compactness", 0.1, 10.0, 1.0, 0.1)
        
        # Parameters for dominant color
        elif method == "Dominant color extraction":
            num_colors = st.slider("Number of colors to extract", 2, 20, 5)
            min_percentage = st.slider("Minimum color percentage", 0.1, 10.0, 1.0, 0.1)
        
        # Parameters for thresholding
        elif method == "Color thresholding":
            threshold_value = st.slider("Threshold sensitivity", 5, 100, 25)
            blur_amount = st.slider("Blur amount", 0, 10, 3)
        
        # Parameters for LAB color space
        elif method == "LAB color space":
            num_colors = st.slider("Number of colors to extract", 2, 20, 5)
            delta_e = st.slider("Color difference threshold (Delta E)", 1, 50, 15)
        
        # Global parameters
        bg_color = st.color_picker("Background color", "#FFFFFF")
        bg_color_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        noise_reduction = st.slider("Noise reduction", 0, 10, 2)
        
        # Post-processing
        st.markdown("<h3>Post-processing</h3>", unsafe_allow_html=True)
        apply_smoothing = st.checkbox("Apply smoothing", True)
        if apply_smoothing:
            smoothing_amount = st.slider("Smoothing amount", 1, 15, 3, 2)
        
        apply_sharpening = st.checkbox("Apply sharpening", False)
        if apply_sharpening:
            sharpening_amount = st.slider("Sharpening amount", 0.1, 5.0, 1.0, 0.1)

# Main content
if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    # Reading the image
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with col1:
        st.markdown("<h2 class='sub-header'>Original Image</h2>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        
        # Image info
        st.markdown("<h3>Image Information</h3>", unsafe_allow_html=True)
        st.write(f"Size: {image.width} x {image.height} pixels")
        st.write(f"Format: {image.format}")
        st.write(f"Mode: {image.mode}")
        
    with col2:
        st.markdown("<h2 class='sub-header'>Separated Color Layers</h2>", unsafe_allow_html=True)
        
        # Apply selected method
        with st.spinner("Separating colors... Please wait."):
            # Process the image based on the selected method
            if method == "K-means clustering":
                color_layers, color_info = kmeans_color_separation(
                    img_cv, 
                    n_colors=num_colors,
                    compactness=compactness,
                    bg_color=bg_color_rgb,
                    noise_reduction=noise_reduction,
                    apply_smoothing=apply_smoothing,
                    smoothing_amount=smoothing_amount if apply_smoothing else 0,
                    apply_sharpening=apply_sharpening,
                    sharpening_amount=sharpening_amount if apply_sharpening else 0
                )
            
            elif method == "Dominant color extraction":
                color_layers, color_info = dominant_color_separation(
                    img_cv, 
                    n_colors=num_colors,
                    min_percentage=min_percentage,
                    bg_color=bg_color_rgb,
                    noise_reduction=noise_reduction,
                    apply_smoothing=apply_smoothing,
                    smoothing_amount=smoothing_amount if apply_smoothing else 0,
                    apply_sharpening=apply_sharpening,
                    sharpening_amount=sharpening_amount if apply_sharpening else 0
                )
            
            elif method == "Color thresholding":
                color_layers, color_info = threshold_color_separation(
                    img_cv, 
                    threshold=threshold_value,
                    blur_amount=blur_amount,
                    bg_color=bg_color_rgb,
                    noise_reduction=noise_reduction,
                    apply_smoothing=apply_smoothing,
                    smoothing_amount=smoothing_amount if apply_smoothing else 0,
                    apply_sharpening=apply_sharpening,
                    sharpening_amount=sharpening_amount if apply_sharpening else 0
                )
            
            elif method == "LAB color space":
                color_layers, color_info = lab_color_separation(
                    img_cv, 
                    n_colors=num_colors,
                    delta_e=delta_e,
                    bg_color=bg_color_rgb,
                    noise_reduction=noise_reduction,
                    apply_smoothing=apply_smoothing,
                    smoothing_amount=smoothing_amount if apply_smoothing else 0,
                    apply_sharpening=apply_sharpening,
                    sharpening_amount=sharpening_amount if apply_sharpening else 0
                )
        
        # Show the extracted layers
        for i, (layer, info) in enumerate(zip(color_layers, color_info)):
            col_left, col_right = st.columns([3, 1])
            
            with col_left:
                # Convert layer from BGR to RGB for display
                layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                st.image(layer_rgb, caption=f"Layer {i+1}", use_column_width=True)
            
            with col_right:
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    info['color'][2], info['color'][1], info['color'][0]  # BGR to RGB
                )
                st.markdown(
                    f"<div><span class='color-chip' style='background-color: {hex_color}'></span> {hex_color}</div>",
                    unsafe_allow_html=True
                )
                st.write(f"RGB: {info['color'][::-1]}")  # BGR to RGB
                st.write(f"Coverage: {info['percentage']:.1f}%")
        
        # Create a combined preview
        if len(color_layers) > 0:
            st.markdown("<h3>Combined Preview</h3>", unsafe_allow_html=True)
            # Create a combined image 
            combined = np.zeros_like(img_cv)
            for layer in color_layers:
                # Take non-black parts of each layer
                mask = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY) > 0
                combined[mask] = layer[mask]
            
            # Convert combined from BGR to RGB for display
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            st.image(combined_rgb, caption="Combined layers", use_column_width=True)
        
        # Download options
        st.markdown("<h3>Download Options</h3>", unsafe_allow_html=True)
        
        if st.button("Prepare Download Package"):
            with st.spinner("Preparing files for download..."):
                # Create a temporary directory
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Save each layer
                    layer_files = []
                    for i, layer in enumerate(color_layers):
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]
                        )
                        layer_filename = f"layer_{i+1}_{hex_color}.png"
                        layer_path = os.path.join(tmpdirname, layer_filename)
                        
                        # Convert BGR to RGB before saving
                        layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                        Image.fromarray(layer_rgb).save(layer_path)
                        layer_files.append(layer_path)
                    
                    # Save combined image
                    combined_path = os.path.join(tmpdirname, "combined.png")
                    Image.fromarray(combined_rgb).save(combined_path)
                    
                    # Create a zip file
                    zip_path = os.path.join(tmpdirname, "color_layers.zip")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for file in layer_files:
                            zipf.write(file, os.path.basename(file))
                        zipf.write(combined_path, os.path.basename(combined_path))
                    
                    # Read the zip file
                    with open(zip_path, "rb") as f:
                        zip_data = f.read()
                    
                    # Provide download link
                    st.download_button(
                        label="Download all layers",
                        data=zip_data,
                        file_name="color_layers.zip",
                        mime="application/zip"
                    )

else:
    # Display sample usage when no image is uploaded
    st.markdown("<h2 class='sub-header'>How to use ColorSep</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    1. Upload an image using the sidebar file uploader
    2. Choose a color separation method:
       - **K-means clustering**: Segments the image into distinct color clusters
       - **Dominant color extraction**: Extracts the most common colors
       - **Color thresholding**: Uses thresholds to separate colors
       - **LAB color space**: Uses perceptual color differences for more accurate separation
    3. Adjust parameters to fine-tune the separation
    4. View each color layer and combined preview
    5. Download individual layers or all layers as a zip file
    
    This tool is ideal for textile printing where each color needs to be printed separately.
    """)
    
    st.info("⬅️ Use the sidebar to upload your image and get started!")
