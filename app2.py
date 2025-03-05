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
    exact_color_separation,
    invert_layer,
    erode_dilate_layer,
    transform_layer,
    adjust_layer_opacity,
    apply_blur_sharpen,
    apply_threshold,
    combine_layers,
    change_layer_color
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
        st.markdown("<h3>Manipulate your color layers here</h3>", unsafe_allow_html=True)
        
        if len(color_layers) > 0:
            # Select a layer to manipulate
            layer_idx = st.selectbox(
                "Select layer to manipulate",
                range(len(color_layers)),
                format_func=lambda i: f"Layer {i+1}: {color_info[i]['color']}"
            )
            
            # Show current layer preview
            st.image(cv2.cvtColor(color_layers[layer_idx], cv2.COLOR_BGR2RGB), caption=f"Original Layer {layer_idx+1}", width=200)
            
            # Create tabs for different manipulation categories
            manipulation_tabs = st.tabs(["Basic", "Morphology", "Transform", "Effects", "Combine Layers", "Change Color"])
            
            # Basic operations (invert, threshold)
            with manipulation_tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Invert Layer", key="invert_layer"):
                        with st.spinner("Inverting layer..."):
                            modified_layer = invert_layer(
                                color_layers[layer_idx],
                                bg_color=bg_color_rgb
                            )
                            st.session_state.custom_layers.append({
                                'layer': modified_layer,
                                'description': f"Inverted Layer {layer_idx+1}"
                            })
                            st.success("Layer inverted successfully!")
                
                with col2:
                    threshold_val = st.slider("Threshold value", 0, 255, 127, key="threshold_slider")
                    if st.button("Apply Threshold", key="apply_threshold"):
                        with st.spinner("Applying threshold..."):
                            modified_layer = apply_threshold(
                                color_layers[layer_idx],
                                threshold_value=threshold_val,
                                bg_color=bg_color_rgb
                            )
                            st.session_state.custom_layers.append({
                                'layer': modified_layer,
                                'description': f"Thresholded Layer {layer_idx+1} (value: {threshold_val})"
                            })
                            st.success("Threshold applied successfully!")
            
            # Morphology operations (erode, dilate)
            with manipulation_tabs[1]:
                morph_op = st.radio("Operation", ["Erode", "Dilate"], key="morph_op")
                col1, col2 = st.columns(2)
                
                with col1:
                    kernel_size = st.slider("Kernel size", 1, 15, 3, 2, key="kernel_size")
                
                with col2:
                    iterations = st.slider("Iterations", 1, 10, 1, key="iterations")
                
                if st.button("Apply Operation", key="apply_morph"):
                    with st.spinner(f"Applying {morph_op.lower()}..."):
                        modified_layer = erode_dilate_layer(
                            color_layers[layer_idx],
                            operation=morph_op.lower(),
                            kernel_size=kernel_size,
                            iterations=iterations,
                            bg_color=bg_color_rgb
                        )
                        st.session_state.custom_layers.append({
                            'layer': modified_layer,
                            'description': f"{morph_op}d Layer {layer_idx+1} (size: {kernel_size}, iter: {iterations})"
                        })
                        st.success(f"{morph_op} operation applied successfully!")
            
            # Transform operations (rotate, flip)
            with manipulation_tabs[2]:
                transform_op = st.selectbox(
                    "Transformation",
                    ["Rotate 90°", "Rotate 180°", "Rotate 270°", "Flip Horizontal", "Flip Vertical"],
                    key="transform_op"
                )
                
                # Create mapping between user-friendly names and function parameters
                transform_map = {
                    "Rotate 90°": "rotate90",
                    "Rotate 180°": "rotate180",
                    "Rotate 270°": "rotate270",
                    "Flip Horizontal": "flip_h",
                    "Flip Vertical": "flip_v"
                }
                
                if st.button("Apply Transform", key="apply_transform"):
                    with st.spinner(f"Applying {transform_op}..."):
                        modified_layer = transform_layer(
                            color_layers[layer_idx],
                            operation=transform_map[transform_op],
                            bg_color=bg_color_rgb
                        )
                        st.session_state.custom_layers.append({
                            'layer': modified_layer,
                            'description': f"Transformed Layer {layer_idx+1} ({transform_op})"
                        })
                        st.success(f"Transform {transform_op} applied successfully!")
            
            # Effects (blur, sharpen, opacity)
            with manipulation_tabs[3]:
                effect_subtabs = st.tabs(["Blur/Sharpen", "Opacity"])
                
                with effect_subtabs[0]:
                    effect_op = st.radio("Effect", ["Blur", "Sharpen"], key="effect_op")
                    effect_amount = st.slider("Amount", 1, 15, 5, key="effect_amount")
                    
                    if st.button("Apply Effect", key="apply_effect"):
                        with st.spinner(f"Applying {effect_op}..."):
                            modified_layer = apply_blur_sharpen(
                                color_layers[layer_idx],
                                operation=effect_op.lower(),
                                amount=effect_amount,
                                bg_color=bg_color_rgb
                            )
                            st.session_state.custom_layers.append({
                                'layer': modified_layer,
                                'description': f"{effect_op}ed Layer {layer_idx+1} (amount: {effect_amount})"
                            })
                            st.success(f"{effect_op} effect applied successfully!")
                
                with effect_subtabs[1]:
                    opacity = st.slider("Opacity", 0.0, 1.0, 0.5, 0.05, key="opacity_slider")
                    
                    if st.button("Apply Opacity", key="apply_opacity"):
                        with st.spinner("Adjusting opacity..."):
                            modified_layer = adjust_layer_opacity(
                                color_layers[layer_idx],
                                opacity=opacity,
                                bg_color=bg_color_rgb
                            )
                            st.session_state.custom_layers.append({
                                'layer': modified_layer,
                                'description': f"Layer {layer_idx+1} with {int(opacity*100)}% opacity"
                            })
                            st.success("Opacity adjusted successfully!")
            
            # Combine Layers
            with manipulation_tabs[4]:
                if len(color_layers) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        layer1_idx = st.selectbox(
                            "Select first layer",
                            range(len(color_layers)),
                            format_func=lambda i: f"Layer {i+1}: {color_info[i]['color']}",
                            key="combine_layer1"
                        )
                    
                    with col2:
                        layer2_idx = st.selectbox(
                            "Select second layer",
                            range(len(color_layers)),
                            format_func=lambda i: f"Layer {i+1}: {color_info[i]['color']}",
                            key="combine_layer2",
                            index=min(1, len(color_layers)-1)  # Select second layer by default
                        )
                    
                    # Option to use a custom color for the combined layer
                    use_custom_color = st.checkbox("Use custom color for combined layer", value=False)
                    
                    if use_custom_color:
                        custom_color_hex = st.color_picker("Pick a color", "#0000FF")
                        r, g, b = int(custom_color_hex[1:3], 16), int(custom_color_hex[3:5], 16), int(custom_color_hex[5:7], 16)
                        custom_color = (b, g, r)  # Convert to BGR
                    else:
                        custom_color = None
                    
                    if st.button("Combine Layers", key="combine_layers_btn"):
                        with st.spinner("Combining layers..."):
                            # Get the selected layers
                            layer1 = color_layers[layer1_idx]
                            layer2 = color_layers[layer2_idx]
                            
                            # Combine the layers
                            combined_layer = combine_layers(
                                layer1,
                                layer2,
                                color=custom_color,
                                bg_color=bg_color_rgb
                            )
                            
                            st.session_state.custom_layers.append({
                                'layer': combined_layer,
                                'description': f"Combined Layer {layer1_idx+1} + Layer {layer2_idx+1}"
                            })
                            st.success("Layers combined successfully!")
                            
                            # Show preview of combined layer
                            st.image(
                                cv2.cvtColor(combined_layer, cv2.COLOR_BGR2RGB),
                                caption=f"Combined Layer Preview (Layer {layer1_idx+1} + Layer {layer2_idx+1})",
                                width=300
                            )
                else:
                    st.warning("You need at least 2 layers to use this feature")
            
            # Change Layer Color
            with manipulation_tabs[5]:
                selected_layer_idx = st.selectbox(
                    "Select layer to recolor",
                    range(len(color_layers)),
                    format_func=lambda i: f"Layer {i+1}: {color_info[i]['color']}",
                    key="recolor_layer"
                )
                
                # Show the current layer
                st.image(
                    cv2.cvtColor(color_layers[selected_layer_idx], cv2.COLOR_BGR2RGB),
                    caption=f"Original Layer {selected_layer_idx+1}",
                    width=200
                )
                
                # Color selection tabs
                color_selection_tabs = st.tabs(["Color Picker", "RGB Values", "Hex Code"])
                
                with color_selection_tabs[0]:
                    # Color picker
                    color_hex = st.color_picker("Pick a color", "#FF0000")
                    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                    new_color = (b, g, r)  # Convert to BGR
                
                with color_selection_tabs[1]:
                    # RGB sliders
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        r_val = st.slider("R", 0, 255, 255, key="r_slider")
                    with col2:
                        g_val = st.slider("G", 0, 255, 0, key="g_slider")
                    with col3:
                        b_val = st.slider("B", 0, 255, 0, key="b_slider")
                    new_color = (b_val, g_val, r_val)  # BGR format
                
                with color_selection_tabs[2]:
                    # Hex code input
                    hex_code = st.text_input("Enter hex color code", "#FF0000")
                    if hex_code:
                        if hex_code[0] != "#":
                            hex_code = "#" + hex_code
                        if len(hex_code) == 7:  # Valid hex code
                            try:
                                r, g, b = int(hex_code[1:3], 16), int(hex_code[3:5], 16), int(hex_code[5:7], 16)
                                new_color = (b, g, r)  # Convert to BGR
                            except ValueError:
                                st.error("Invalid hex code format. Use #RRGGBB format.")
                
                # Show the selected color
                st.markdown(
                    f"<div><span style='background-color: #{new_color[2]:02x}{new_color[1]:02x}{new_color[0]:02x}; width: 50px; height: 30px; display: inline-block; margin-right: 10px;'></span> Selected color: RGB({new_color[2]}, {new_color[1]}, {new_color[0]})</div>",
                    unsafe_allow_html=True
                )
                
                if st.button("Apply New Color", key="change_color_btn"):
                    with st.spinner("Changing layer color..."):
                        # Change the color of the layer
                        recolored_layer = change_layer_color(
                            color_layers[selected_layer_idx],
                            new_color,
                            bg_color=bg_color_rgb
                        )
                        
                        st.session_state.custom_layers.append({
                            'layer': recolored_layer,
                            'description': f"Recolored Layer {selected_layer_idx+1} (RGB: {new_color[2]}, {new_color[1]}, {new_color[0]})"
                        })
                        st.success("Layer color changed successfully!")
                        
                        # Show preview of recolored layer
                        st.image(
                            cv2.cvtColor(recolored_layer, cv2.COLOR_BGR2RGB),
                            caption=f"Recolored Layer {selected_layer_idx+1}",
                            width=200
                        )
            
            # Show instruction for finding manipulated layers
            if len(st.session_state.custom_layers) > 0:
                st.info("Your manipulated layers are available below:")
                
                # Display the manipulated layers gallery
                st.markdown("<h3>Manipulated Layers Gallery</h3>", unsafe_allow_html=True)
                
                # Create a grid layout for manipulated layers
                gallery_cols = st.columns(3)
                
                for i, custom_layer in enumerate(st.session_state.custom_layers):
                    with gallery_cols[i % 3]:
                        # Convert from BGR to RGB for display
                        layer_rgb = cv2.cvtColor(custom_layer['layer'], cv2.COLOR_BGR2RGB)
                        
                        # Display the layer with its description
                        st.image(layer_rgb, caption=custom_layer['description'])
                        
                        # Add download button
                        layer_rgb_pil = Image.fromarray(layer_rgb)
                        buffer = io.BytesIO()
                        layer_rgb_pil.save(buffer, format="PNG")
                        st.download_button(
                            label=f"Download Layer",
                            data=buffer.getvalue(),
                            file_name=f"modified_layer_{i+1}.png",
                            mime="image/png",
                        )
                
                # Add button to clear custom layers
                if st.button("Clear All Modified Layers"):
                    st.session_state.custom_layers = []
                    st.success("All modified layers cleared!")
                    st.experimental_rerun()
        else:
            st.warning("No layers available to manipulate. Please create color layers first.")

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
