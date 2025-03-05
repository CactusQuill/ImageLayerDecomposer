"""
ColorSep: Textile Color Separation Tool with Pantone Color Extraction.
"""

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
import pantone_colors as pantone
from pantone_tab import pantone_extraction_tab
from color_separation import (
    kmeans_color_separation,
    dominant_color_separation,
    threshold_color_separation,
    lab_color_separation,
    exact_color_separation
)

# Set page configuration
st.set_page_config(
    page_title="ColorSep - Textile Color Separation Tool",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set theme settings - light theme for better text visibility
st.markdown("""
    <script>
        var elements = window.parent.document.querySelectorAll('.stApp')
        elements[0].style.backgroundColor = '#ffffff';
    </script>
    """, unsafe_allow_html=True)

# Custom CSS (same as in original app)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0056b3;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #212121;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .info-text {
        font-size: 1.1rem;
        color: #000000;
        line-height: 1.5;
    }
    /* Other CSS styles from the original app... */
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>ColorSep: Textile Color Separation Tool</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Upload an image and extract different color layers for textile printing</p>", unsafe_allow_html=True)

# Import Pantone color codes
pantone_codes = pantone.get_all_pantone_codes()

# Sidebar for controls (same as in original app)
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    
    # Only show settings if an image is uploaded
    if uploaded_file is not None:
        # Color separation settings
        st.markdown("<h3>Color Separation Settings</h3>", unsafe_allow_html=True)
        
        # Method selection
        method = st.selectbox(
            "Color Separation Method",
            [
                "Exact color extraction",
                "K-means clustering",
                "Dominant color extraction",
                "Color thresholding",
                "LAB color space"
            ]
        )
        
        # Parameters for Exact color extraction
        if method == "Exact color extraction":
            max_colors = st.slider("Maximum number of colors to extract", 5, 15, 10)
            st.warning("Note: Images with gradients or noise may have many unique colors. This method creates one layer per unique color.")
            
        # Parameters for K-means
        elif method == "K-means clustering":
            num_colors = st.slider("Number of colors", 2, 15, 5)
            compactness = st.slider("Color compactness", 0.1, 10.0, 1.0, 0.1)
            st.info("Higher compactness values create more distinct color boundaries.")
            
        # Parameters for Dominant color extraction
        elif method == "Dominant color extraction":
            num_colors = st.slider("Number of colors", 2, 15, 5)
            color_tol = st.slider("Color tolerance", 1, 100, 20)
            st.info("Lower tolerance values create more precise color matching.")
            
        # Parameters for Color thresholding
        elif method == "Color thresholding":
            threshold = st.slider("Threshold", 10, 100, 30)
            st.info("Lower threshold values extract more colors but may include noise.")
            
        # Parameters for LAB color space
        elif method == "LAB color space":
            lab_distance = st.slider("Color distance", 1, 50, 15)
            st.info("Lower distance values create more precise color separation but may miss similar shades.")
        
        # Background color option
        st.markdown("<h3>Background Settings</h3>", unsafe_allow_html=True)
        bg_color = st.selectbox(
            "Background Color",
            ["White", "Black", "Transparent"],
            index=2
        )
        
        # Pre-processing options
        st.markdown("<h3>Pre-processing Options</h3>", unsafe_allow_html=True)
        apply_blur = st.checkbox("Apply blur (reduces noise)", value=False)
        if apply_blur:
            blur_amount = st.slider("Blur amount", 1, 15, 3, 2)
            
        apply_edge_preserve = st.checkbox("Edge-preserving smoothing", value=False)
        if apply_edge_preserve:
            edge_preserve_strength = st.slider("Strength", 10, 100, 30)
            
        apply_sharpening = st.checkbox("Apply sharpening", value=False)
        if apply_sharpening:
            sharpening_amount = st.slider("Sharpening amount", 0.1, 5.0, 1.0, 0.1)

# Create tabs for different functionality
tab1, tab2, tab3, tab4 = st.tabs([
    "Color Separation", "Layer Manipulation", "Pantone Extraction", "Help"
])

# Tab 1: Color Separation
with tab1:
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
            # Add more image information as needed
        
        with col2:
            st.markdown("<h2 class='sub-header'>Separated Color Layers</h2>", unsafe_allow_html=True)
            
            # Apply selected method
            with st.spinner("Separating colors... Please wait."):
                # Pre-processing
                processed_img = img_cv.copy()
                
                if apply_blur:
                    processed_img = cv2.GaussianBlur(processed_img, (blur_amount, blur_amount), 0)
                
                if apply_edge_preserve:
                    processed_img = cv2.edgePreservingFilter(processed_img, flags=1, sigma_s=60, sigma_r=edge_preserve_strength/100)
                
                if apply_sharpening:
                    blurred = cv2.GaussianBlur(processed_img, (5, 5), 0)
                    processed_img = cv2.addWeighted(processed_img, 1 + sharpening_amount, blurred, -sharpening_amount, 0)
                
                # Set background color
                if bg_color == "White":
                    bg_color_rgb = (255, 255, 255)
                elif bg_color == "Black":
                    bg_color_rgb = (0, 0, 0)
                else:  # Transparent
                    bg_color_rgb = (0, 0, 0)  # Will be made transparent later
            
                # Color separation based on method
                if method == "Exact color extraction":
                    color_layers, color_info = exact_color_separation(processed_img, max_colors, bg_color_rgb)
                
                elif method == "K-means clustering":
                    color_layers, color_info = kmeans_color_separation(processed_img, num_colors, compactness, bg_color_rgb)
                
                elif method == "Dominant color extraction":
                    color_layers, color_info = dominant_color_separation(processed_img, num_colors, color_tol, bg_color_rgb)
                
                elif method == "Color thresholding":
                    color_layers, color_info = threshold_color_separation(processed_img, threshold, bg_color_rgb)
                
                elif method == "LAB color space":
                    color_layers, color_info = lab_color_separation(processed_img, lab_distance, bg_color_rgb)
                
                # Initialize session state variables for layer ordering if needed
                if 'layer_order' not in st.session_state:
                    st.session_state.layer_order = list(range(len(color_layers)))
                    
                if 'layer_visibility' not in st.session_state:
                    st.session_state.layer_visibility = [True] * len(color_layers)
                    
                if 'custom_layers' not in st.session_state:
                    st.session_state.custom_layers = []
            
            # Display a gallery of color layers
            if len(color_layers) > 0:
                st.success(f"Successfully extracted {len(color_layers)} color layers")
                st.markdown("<h3>Color Layers</h3>", unsafe_allow_html=True)
                
                # Create a grid layout
                cols = st.columns(3)
                
                for i, layer in enumerate(color_layers):
                    with cols[i % 3]:
                        # Convert from BGR to RGB for display
                        layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                        
                        # Calculate the percentage of this color
                        percentage = color_info[i]['percentage']
                        color_value = color_info[i]['color']
                        
                        # Create hex code for display
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            color_value[2], color_value[1], color_value[0]
                        )
                        
                        st.image(layer_rgb, caption=f"Layer {i+1}: {percentage:.1f}%")
                        st.markdown(
                            f"<div><span style='background-color: {hex_color}; width: 20px; height: 20px; display: inline-block; margin-right: 5px;'></span> Color: {hex_color}</div>",
                            unsafe_allow_html=True
                        )
                        
                        # Add download button for each layer
                        layer_rgb_pil = Image.fromarray(layer_rgb)
                        buffer = io.BytesIO()
                        layer_rgb_pil.save(buffer, format="PNG")
                        st.download_button(
                            label=f"Download Layer {i+1}",
                            data=buffer.getvalue(),
                            file_name=f"layer_{i+1}_{hex_color[1:]}.png",
                            mime="image/png",
                        )
    else:
        st.info("Please upload an image in the sidebar to get started.")

# Tab 2: Layer Manipulation
with tab2:
    st.header("Layer Manipulation")
    
    if uploaded_file is None:
        st.info("Please upload an image in the sidebar first")
    elif 'color_layers' not in locals():
        st.info("Please go to the Color Separation tab first to extract colors")
    else:
        # Add your layer manipulation code here
        st.write("Manipulate your color layers here")

# Tab 3: Pantone Extraction
with tab3:
    pantone_extraction_tab()

# Tab 4: Help
with tab4:
    st.header("How to use ColorSep")
    st.markdown("""
    ColorSep is a tool for textile printing color separation. It extracts different color layers from an image, which is useful for creating separate screens in textile printing processes.
    
    ### Features:
    - Multiple color separation methods
    - Advanced layer manipulation
    - Pantone color matching and extraction
    - Download options for individual layers or complete packages
    
    This tool is ideal for textile printing where each color needs to be printed separately.
    
    ### Advanced Features:
    - **Combine Layers**: Merge two color layers into a single layer
    - **Change Layer Colors**: Modify the color of any layer using RGB, Hex, or Pantone color codes
    - **Pantone Matching**: Extract layers that match specific Pantone TPG/TPX colors
    """)
    
    st.info("⬅️ Use the sidebar to upload your image and get started!")
