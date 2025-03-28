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

# Custom CSS
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
    .stButton button {
        background-color: #0056b3;
        color: white;
        font-weight: bold;
    }
    .color-chip {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 5px;
        border: 1px solid #000;
    }
    /* Improve general text visibility */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, label, .stSelectbox, .stSlider {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    /* Improve contrast for selectbox and slider labels */
    .stSelectbox label, .stSlider label {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    /* Add background contrast to important sections */
    .stExpander {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>ColorSep: Textile Color Separation Tool</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Upload an image and extract different color layers for textile printing</p>", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'custom_layers' not in st.session_state:
    st.session_state.custom_layers = []
    
if 'layer_visibility' not in st.session_state:
    st.session_state.layer_visibility = []
    
if 'layer_order' not in st.session_state:
    st.session_state.layer_order = []

# Import color separation functions from external modules
from color_separation import (
    kmeans_color_separation,
    dominant_color_separation, 
    threshold_color_separation,
    lab_color_separation,
    exact_color_separation,
    combine_layers,
    change_layer_color,
    get_color_from_code,
    invert_layer,
    erode_dilate_layer,
    transform_layer,
    adjust_layer_opacity,
    apply_blur_sharpen,
    apply_threshold
)

# Import Pantone color codes
pantone_codes = pantone.get_all_pantone_codes()

# Sidebar for controls
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        method = st.selectbox(
            "Choose separation method",
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
            if method == "Exact color extraction":
                color_layers, color_info = exact_color_separation(
                    img_cv,
                    max_colors=max_colors,
                    bg_color=bg_color_rgb
                )
            
            elif method == "K-means clustering":
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
                
                # Add download button for this individual layer
                layer_rgb_pil = Image.fromarray(layer_rgb)
                layer_bytes = io.BytesIO()
                layer_rgb_pil.save(layer_bytes, format="PNG")
                
                hex_color = "{:02x}{:02x}{:02x}".format(
                    info['color'][2], info['color'][1], info['color'][0]  # BGR to RGB
                )
                
                st.download_button(
                    label=f"Download Layer {i+1}",
                    data=layer_bytes.getvalue(),
                    file_name=f"layer_{i+1}_{hex_color}.png",
                    mime="image/png",
                    key=f"download_layer_{i}"
                )
            
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
            st.markdown("""
            <div style='background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                <h3>Combined Preview</h3>
                <p>Control the visual order of layers in your preview. Use the Layer Order & Visibility Settings below to:</p>
                <ul>
                    <li>Change the stacking order of layers (higher position numbers appear on top)</li>
                    <li>Toggle layer visibility on/off to preview different combinations</li>
                    <li>Save your current layer arrangement for further editing</li>
                </ul>
                <p>All downloads will respect your layer order and visibility settings.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add an option to reorder layers
            with st.expander("Layer Order & Visibility Settings"):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write("Control the stacking order and visibility of your layers:")
                
                with col2:
                    # Add a button to reset all layer visibility
                    if st.button("Show All Layers"):
                        st.session_state.layer_visibility = [True] * len(color_layers)
                        st.experimental_rerun()
                
                # Create a list of layer indices and their names for sorting
                layer_items = [f"Layer {i+1} - {color_info[i]['percentage']:.1f}% - #{color_info[i]['color'][2]:02x}{color_info[i]['color'][1]:02x}{color_info[i]['color'][0]:02x}" 
                              for i in range(len(color_layers))]
                
                # Store the current order if not already in session state
                if 'layer_order' not in st.session_state:
                    st.session_state.layer_order = list(range(len(color_layers)))
                # Update layer_order if number of layers changed
                elif len(st.session_state.layer_order) != len(color_layers):
                    st.session_state.layer_order = list(range(len(color_layers)))
                
                # Initialize layer visibility if not in session state
                if 'layer_visibility' not in st.session_state:
                    st.session_state.layer_visibility = [True] * len(color_layers)
                # Update visibility if number of layers changed
                elif len(st.session_state.layer_visibility) != len(color_layers):
                    st.session_state.layer_visibility = [True] * len(color_layers)
                
                # Create columns for each layer to allow reordering and toggling visibility
                for i in range(len(color_layers)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        order_value = st.number_input(
                            f"Layer {i+1} position",
                            min_value=1,
                            max_value=len(color_layers),
                            value=st.session_state.layer_order[i] + 1,
                            key=f"layer_order_{i}"
                        )
                        st.session_state.layer_order[i] = order_value - 1
                    
                    with col2:
                        st.session_state.layer_visibility[i] = st.checkbox(
                            f"Visible",
                            value=st.session_state.layer_visibility[i],
                            key=f"layer_visible_{i}"
                        )
                    
                    # Display a color swatch for this layer
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]  # BGR to RGB
                    )
                    st.markdown(
                        f"<div><span class='color-chip' style='background-color: {hex_color}; width: 100%; height: 10px;'></span></div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
            
            # Create a combined image based on the user-defined order
            combined = np.zeros_like(img_cv)
            
            # Apply layers in reverse order (so that last layer is on top)
            for idx in reversed(st.session_state.layer_order):
                if st.session_state.layer_visibility[idx]:
                    layer = color_layers[idx]
                    # Take non-black parts of each layer
                    mask = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY) > 0
                    combined[mask] = layer[mask]
            
            # Convert combined from BGR to RGB for display
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            
            # Count visible layers
            visible_layers = sum(st.session_state.layer_visibility)
            total_layers = len(color_layers)
            
            # Create a caption that shows layer order status
            if visible_layers == total_layers:
                caption = f"Combined preview of all {total_layers} layers with custom ordering"
            else:
                caption = f"Combined preview of {visible_layers}/{total_layers} visible layers with custom ordering"
                
            st.image(combined_rgb, caption=caption, use_column_width=True)
            
            # Download button for combined preview
            combined_rgb_pil = Image.fromarray(combined_rgb)
            combined_bytes = io.BytesIO()
            combined_rgb_pil.save(combined_bytes, format="PNG")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Combined Preview",
                    data=combined_bytes.getvalue(),
                    file_name="combined_preview.png",
                    mime="image/png",
                    key="download_combined_preview"
                )
            
            with col2:
                # Create a button to save the layer order
                if st.button("Save Current Layer Order", key="save_layer_order"):
                    if 'custom_layers' not in st.session_state:
                        st.session_state.custom_layers = []
                    
                    # Add the ordered combined image to custom layers
                    st.session_state.custom_layers.append({
                        'layer': combined,
                        'name': f"Combined with custom order"
                    })
                    
                    st.success("Current layer order saved to Manipulated Layers Gallery!")
        
        # Download options
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h3>Download Options</h3>
            <p>Choose from different download formats to suit your textile printing workflow.</p>
        </div>
        """, unsafe_allow_html=True)
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            if st.button("Prepare All Layers Package"):
                with st.spinner("Preparing files for download..."):
                    # Create a temporary directory
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        # Determine if we have an ordered layer list
                        has_ordered_layers = 'layer_order' in st.session_state and len(st.session_state.layer_order) == len(color_layers)
                        has_visibility = 'layer_visibility' in st.session_state and len(st.session_state.layer_visibility) == len(color_layers)
                        
                        # Save each layer with order information
                        layer_files = []
                        for i, layer in enumerate(color_layers):
                            # Get the layer's position in the stack (if ordering is active)
                            position = i
                            if has_ordered_layers:
                                # Find this layer's position in the ordered list
                                position = st.session_state.layer_order[i]
                                
                            # Skip layers that are set to be invisible
                            if has_visibility and not st.session_state.layer_visibility[i]:
                                continue
                            
                            hex_color = "{:02x}{:02x}{:02x}".format(
                                color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]
                            )
                            # Include position in filename
                            layer_filename = f"position{position+1:02d}_layer_{i+1}_{hex_color}.png"
                            layer_path = os.path.join(tmpdirname, layer_filename)
                            
                            # Convert BGR to RGB before saving
                            layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                            Image.fromarray(layer_rgb).save(layer_path)
                            layer_files.append(layer_path)
                        
                        # Save combined image
                        combined_path = os.path.join(tmpdirname, "combined.png")
                        Image.fromarray(combined_rgb).save(combined_path)
                        
                        # Create a README text file explaining the layers
                        readme_content = """# ColorSep Exported Layers

This package contains color-separated layers for textile printing.

## File Naming Convention
- Files are named with the following pattern: position{XX}_layer_{Y}_{color}.png
- Position: The position in the stacking order (01 is the bottom layer, higher numbers are on top)
- Layer: The original layer number from the extraction
- Color: The hex color code of the layer

## Contents
- Combined image: The combined result of all layers according to their stacking order
"""
                        # Add information about each layer to the README
                        readme_content += "\n\n## Layer Details\n"
                        for i, layer in enumerate(color_layers):
                            position = i
                            if 'layer_order' in st.session_state and len(st.session_state.layer_order) == len(color_layers):
                                position = st.session_state.layer_order[i]
                            
                            hex_color = "{:02x}{:02x}{:02x}".format(
                                color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]
                            )
                            
                            readme_content += f"- Layer {i+1}: Position {position+1}, Coverage {color_info[i]['percentage']:.1f}%, Color #{hex_color}\n"
                        
                        # Save README file
                        readme_path = os.path.join(tmpdirname, "README.txt")
                        with open(readme_path, 'w') as f:
                            f.write(readme_content)
                        
                        # Create a zip file
                        zip_path = os.path.join(tmpdirname, "color_layers.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in layer_files:
                                zipf.write(file, os.path.basename(file))
                            zipf.write(combined_path, os.path.basename(combined_path))
                            zipf.write(readme_path, os.path.basename(readme_path))
                        
                        # Read the zip file
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                        
                        # Provide download link
                        st.download_button(
                            label="Download All Layers",
                            data=zip_data,
                            file_name="color_layers.zip",
                            mime="application/zip",
                            key="download_all_zip"
                        )
                        
        with download_col2:
            if st.button("Save as PNG Masks"):
                with st.spinner("Preparing mask files for download..."):
                    # Create a temporary directory
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        # Determine if we have an ordered layer list
                        has_ordered_layers = 'layer_order' in st.session_state and len(st.session_state.layer_order) == len(color_layers)
                        has_visibility = 'layer_visibility' in st.session_state and len(st.session_state.layer_visibility) == len(color_layers)
                        
                        # Save each layer as a black and white mask
                        mask_files = []
                        for i, layer in enumerate(color_layers):
                            # Get the layer's position in the stack (if ordering is active)
                            position = i
                            if has_ordered_layers:
                                # Find this layer's position in the ordered list
                                position = st.session_state.layer_order[i]
                                
                            # Skip layers that are set to be invisible
                            if has_visibility and not st.session_state.layer_visibility[i]:
                                continue
                            
                            # Create mask (white foreground, black background)
                            mask = np.zeros((layer.shape[0], layer.shape[1]), dtype=np.uint8)
                            is_fg = np.logical_not(np.all(layer == bg_color_rgb, axis=2))
                            mask[is_fg] = 255
                            
                            hex_color = "{:02x}{:02x}{:02x}".format(
                                color_info[i]['color'][2], color_info[i]['color'][1], color_info[i]['color'][0]
                            )
                            # Include position in filename
                            mask_filename = f"position{position+1:02d}_mask_{i+1}_{hex_color}.png"
                            mask_path = os.path.join(tmpdirname, mask_filename)
                            
                            # Save the mask
                            Image.fromarray(mask).save(mask_path)
                            mask_files.append(mask_path)
                        
                        # Create a zip file
                        zip_path = os.path.join(tmpdirname, "mask_layers.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in mask_files:
                                zipf.write(file, os.path.basename(file))
                        
                        # Read the zip file
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                        
                        # Provide download link
                        st.download_button(
                            label="Download Mask Layers",
                            data=zip_data,
                            file_name="mask_layers.zip",
                            mime="application/zip",
                            key="download_masks_zip"
                        )
        
        # Layer manipulation tools
        st.markdown("""
        <div style='background-color: #f2f8f3; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
            <h3>Layer Manipulation Tools</h3>
            <p>Combine layers or change their colors to achieve the perfect separation for your textile printing project.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Combine Layers"):
            if len(color_layers) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    layer1_idx = st.selectbox(
                        "Select first layer",
                        range(len(color_layers)),
                        format_func=lambda i: f"Layer {i+1} - {color_info[i]['percentage']:.1f}%"
                    )
                with col2:
                    layer2_idx = st.selectbox(
                        "Select second layer",
                        range(len(color_layers)),
                        format_func=lambda i: f"Layer {i+1} - {color_info[i]['percentage']:.1f}%",
                        index=min(1, len(color_layers)-1)  # Default to second layer
                    )
                
                use_custom_color = st.checkbox("Use custom color for combined layer")
                custom_color = None
                
                if use_custom_color:
                    color_input_method = st.radio(
                        "Color input method",
                        ["Color Picker", "RGB Value", "Hex Code", "Pantone TPX/TPG"],
                        horizontal=True
                    )
                    
                    if color_input_method == "Color Picker":
                        custom_color_hex = st.color_picker("Select color", "#FF0000")
                        custom_color = pantone.get_color_from_code(custom_color_hex)
                    
                    elif color_input_method == "RGB Value":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            r_val = st.number_input("R", 0, 255, 255)
                        with col2:
                            g_val = st.number_input("G", 0, 255, 0)
                        with col3:
                            b_val = st.number_input("B", 0, 255, 0)
                        custom_color = (b_val, g_val, r_val)  # BGR format for OpenCV
                    
                    elif color_input_method == "Hex Code":
                        hex_val = st.text_input("Hex Code (e.g., #FF0000)", "#FF0000")
                        custom_color = pantone.get_color_from_code(hex_val)
                    
                    elif color_input_method == "Pantone TPX/TPG":
                        pantone_code_type = st.selectbox(
                            "Select Pantone code type",
                            ["TPX", "TPG"],
                            key="pantone_code_type"
                        )
                        
                        if pantone_code_type == "TPX":
                            # Create a dictionary of TPX codes and names
                            tpx_codes = {code_info['code']: name for name, code_info in pantone.TPX_COLORS.items()}
                            pantone_tpx_code = st.selectbox(
                                "Select Pantone TPX code",
                                list(tpx_codes.keys()),
                                format_func=lambda x: f"{x} - {tpx_codes[x]}",
                                key="pantone_tpx_code"
                            )
                            # Get color from TPX database
                            selected_color = pantone.TPX_COLORS[tpx_codes[pantone_tpx_code]]['rgb']
                            custom_color = (selected_color[2], selected_color[1], selected_color[0])  # Convert to BGR
                            color_source = "pantone_tpx"
                        elif pantone_code_type == "TPG":
                            # Create a dictionary of TPG codes and names
                            tpg_codes = {code_info['code']: name for name, code_info in pantone.TPG_COLORS.items()}
                            pantone_tpg_code = st.selectbox(
                                "Select Pantone TPG code",
                                list(tpg_codes.keys()),
                                format_func=lambda x: f"{x} - {tpg_codes[x]}",
                                key="pantone_tpg_code"
                            )
                            # Get color from TPG database
                            selected_color = pantone.TPG_COLORS[tpg_codes[pantone_tpg_code]]['rgb']
                            custom_color = (selected_color[2], selected_color[1], selected_color[0])  # Convert to BGR
                            color_source = "pantone_tpg"
                
                if st.button("Combine Layers"):
                    with st.spinner("Combining layers..."):
                        # Get the selected layers
                        layer1 = color_layers[layer1_idx]
                        layer2 = color_layers[layer2_idx]
                        
                        # Combine the layers
                        combined = combine_layers(layer1, layer2, custom_color, bg_color_rgb)
                        
                        # Calculate the percentage of the combined layer
                        h, w = combined.shape[:2]
                        mask = np.zeros((h, w), dtype=np.uint8)
                        is_fg = np.logical_not(np.all(combined == bg_color_rgb, axis=2))
                        mask[is_fg] = 255
                        percentage = (np.sum(mask) / 255 / (h * w)) * 100
                        
                        # Set the color for the combined layer
                        if custom_color:
                            new_color = custom_color
                        else:
                            # Use color from layer1 if no custom color
                            new_color = color_info[layer1_idx]['color']
                        
                        # Remove the original layers
                        replaced_indices = sorted([layer1_idx, layer2_idx], reverse=True)
                        for idx in replaced_indices:
                            color_layers.pop(idx)
                            color_info.pop(idx)
                        
                        # Add the combined layer
                        color_layers.append(combined)
                        color_info.append({
                            'color': new_color,
                            'percentage': percentage
                        })
                        
                        # Display the result
                        result_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, caption="Combined Layer", use_column_width=True)
                        
                        # Add download button for this combined layer
                        result_rgb_pil = Image.fromarray(result_rgb)
                        result_bytes = io.BytesIO()
                        result_rgb_pil.save(result_bytes, format="PNG")
                        
                        hex_color = "{:02x}{:02x}{:02x}".format(
                            new_color[2], new_color[1], new_color[0]  # BGR to RGB
                        )
                        
                        st.download_button(
                            label=f"Download Combined Layer",
                            data=result_bytes.getvalue(),
                            file_name=f"combined_layer_{layer1_idx+1}_{layer2_idx+1}.png",
                            mime="image/png",
                            key=f"download_combined_{layer1_idx}_{layer2_idx}"
                        )
                        
                        # Store this new layer in session state
                        if 'custom_layers' not in st.session_state:
                            st.session_state.custom_layers = []
                        
                        st.session_state.custom_layers.append({
                            'layer': combined,
                            'name': f"Combined {layer1_idx+1} & {layer2_idx+1}"
                        })
                        
                        st.success(f"Layers {layer1_idx+1} and {layer2_idx+1} combined successfully!")
            else:
                st.warning("You need at least 2 layers to use this feature")
        
        with st.expander("Change Layer Color"):
            if len(color_layers) > 0:
                # Select layer to change
                layer_idx = st.selectbox(
                    "Select layer to recolor",
                    range(len(color_layers)),
                    format_func=lambda i: f"Layer {i+1} - {color_info[i]['percentage']:.1f}%"
                )
                
                # Color input method
                color_input_method = st.radio(
                    "Color input method",
                    ["Color Picker", "RGB Value", "Hex Code", "Pantone TPX/TPG"],
                    horizontal=True,
                    key="recolor_method"
                )
                
                new_color = None
                color_source = None
                pantone_tpx_code = None
                pantone_tpg_code = None
                
                if color_input_method == "Color Picker":
                    # Get current color in hex
                    current_color = color_info[layer_idx]['color']
                    current_hex = "#{:02x}{:02x}{:02x}".format(
                        current_color[2], current_color[1], current_color[0]
                    )
                    new_color_hex = st.color_picker("Select new color", current_hex)
                    new_color = pantone.get_color_from_code(new_color_hex)
                    color_source = "color_picker"
                
                elif color_input_method == "RGB Value":
                    col1, col2, col3 = st.columns(3)
                    # Get current color
                    current_color = color_info[layer_idx]['color']
                    
                    with col1:
                        r_val = st.number_input("R", 0, 255, current_color[2])
                    with col2:
                        g_val = st.number_input("G", 0, 255, current_color[1])
                    with col3:
                        b_val = st.number_input("B", 0, 255, current_color[0])
                    new_color = (b_val, g_val, r_val)  # BGR format for OpenCV
                    color_source = "rgb_value"
                
                elif color_input_method == "Hex Code":
                    current_color = color_info[layer_idx]['color']
                    current_hex = "#{:02x}{:02x}{:02x}".format(
                        current_color[2], current_color[1], current_color[0]
                    )
                    hex_val = st.text_input("Hex Code (e.g., #FF0000)", current_hex)
                    new_color = pantone.get_color_from_code(hex_val)
                    color_source = "hex_code"
                
                elif color_input_method == "Pantone TPX/TPG":
                    pantone_code_type = st.selectbox(
                        "Select Pantone code type",
                        ["TPX", "TPG"],
                        key="pantone_code_type"
                    )
                    
                    if pantone_code_type == "TPX":
                        # Create a dictionary of TPX codes and names
                        tpx_codes = {code_info['code']: name for name, code_info in pantone.TPX_COLORS.items()}
                        pantone_tpx_code = st.selectbox(
                            "Select Pantone TPX code",
                            list(tpx_codes.keys()),
                            format_func=lambda x: f"{x} - {tpx_codes[x]}",
                            key="pantone_tpx_code"
                        )
                        # Get color from TPX database
                        selected_color = pantone.TPX_COLORS[tpx_codes[pantone_tpx_code]]['rgb']
                        new_color = (selected_color[2], selected_color[1], selected_color[0])  # Convert to BGR
                        color_source = "pantone_tpx"
                    elif pantone_code_type == "TPG":
                        # Create a dictionary of TPG codes and names
                        tpg_codes = {code_info['code']: name for name, code_info in pantone.TPG_COLORS.items()}
                        pantone_tpg_code = st.selectbox(
                            "Select Pantone TPG code",
                            list(tpg_codes.keys()),
                            format_func=lambda x: f"{x} - {tpg_codes[x]}",
                            key="pantone_tpg_code"
                        )
                        # Get color from TPG database
                        selected_color = pantone.TPG_COLORS[tpg_codes[pantone_tpg_code]]['rgb']
                        new_color = (selected_color[2], selected_color[1], selected_color[0])  # Convert to BGR
                        color_source = "pantone_tpg"
                
                # Preview the color
                st.markdown(
                    f"<div><span class='color-chip' style='background-color: #{new_color[2]:02x}{new_color[1]:02x}{new_color[0]:02x}; width: 50px; height: 30px;'></span> Selected color: RGB({new_color[2]}, {new_color[1]}, {new_color[0]})</div>",
                    unsafe_allow_html=True
                )
                
                # Apply button
                if st.button("Apply New Color"):
                    with st.spinner("Changing layer color..."):
                        # Get the selected layer
                        layer = color_layers[layer_idx]
                        
                        # Change the color
                        recolored_layer = change_layer_color(layer, new_color, bg_color_rgb)
                        
                        # Display results
                        recolored_rgb = cv2.cvtColor(recolored_layer, cv2.COLOR_BGR2RGB) 
                        st.image(recolored_rgb, caption=f"Layer {layer_idx+1} with new color", use_column_width=True)
                        
                        # Add download button for this recolored layer
                        recolored_rgb_pil = Image.fromarray(recolored_rgb)
                        recolored_bytes = io.BytesIO()
                        recolored_rgb_pil.save(recolored_bytes, format="PNG")
                        
                        hex_color = "{:02x}{:02x}{:02x}".format(
                            new_color[2], new_color[1], new_color[0]  # BGR to RGB
                        )
                        
                        st.download_button(
                            label=f"Download Recolored Layer",
                            data=recolored_bytes.getvalue(),
                            file_name=f"recolored_layer_{layer_idx+1}_{hex_color}.png",
                            mime="image/png",
                            key=f"download_recolored_{layer_idx}_{hex_color}"
                        )
                        
                        # Update the color layer and info
                        color_layers[layer_idx] = recolored_layer
                        color_info[layer_idx]['color'] = new_color
                        
                        # Store this recolored layer in session state
                        if 'custom_layers' not in st.session_state:
                            st.session_state.custom_layers = []
                        
                        # Convert BGR color to hex for name
                        hex_color = "{:02x}{:02x}{:02x}".format(
                            new_color[2], new_color[1], new_color[0]  # BGR to RGB
                        )
                        
                        # Get the color name for display if it was selected from a predefined system
                        color_description = f"#{hex_color}"
                        if color_source == "pantone_tpx" and pantone_tpx_code:
                            # Get the Pantone TPX name from the code
                            for name, code_info in pantone.TPX_COLORS.items():
                                if code_info['code'] == pantone_tpx_code:
                                    color_description = f"Pantone TPX {pantone_tpx_code} ({name})"
                                    break
                        elif color_source == "pantone_tpg" and pantone_tpg_code:
                            # Get the Pantone TPG name from the code
                            for name, code_info in pantone.TPG_COLORS.items():
                                if code_info['code'] == pantone_tpg_code:
                                    color_description = f"Pantone TPG {pantone_tpg_code} ({name})"
                                    break
                        
                        st.session_state.custom_layers.append({
                            'layer': recolored_layer,
                            'name': f"Layer {layer_idx+1} recolored to {color_description}"
                        })
                        
                        # Success message
                        st.success(f"Layer {layer_idx+1} has been recolored!")
            else:
                st.warning("No layers available to recolor")

        with st.expander("Layer Manipulation"):
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
                manipulation_tabs = st.tabs(["Basic", "Morphology", "Transform", "Effects"])
                
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
                
                # Initialize session state for custom layers if not already present
                if 'custom_layers' not in st.session_state:
                    st.session_state.custom_layers = []
                
                # Show instruction for finding manipulated layers
                if len(st.session_state.custom_layers) > 0:
                    st.info("👇 Your manipulated layers are available in the 'Manipulated Layers Gallery' section below")
            else:
                st.warning("No layers available to manipulate. Please create color layers first.")
        
        # Add a section to show custom manipulated layers
        if 'custom_layers' in st.session_state and len(st.session_state.custom_layers) > 0:
            st.markdown("""
            <div style='background-color: #f0f7ff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                <h3>Manipulated Layers Gallery</h3>
                <p>Browse and download your custom combined and recolored layers from this session.</p>
            </div>
            """, unsafe_allow_html=True)
            # Create a list of all layers names for the selector
            layer_names = [layer_info['name'] for layer_info in st.session_state.custom_layers]
            
            # Select a layer to view
            selected_layer_name = st.selectbox(
                "Select a manipulated layer to view",
                layer_names,
                key="custom_layer_selector"
            )
            
            # Find the selected layer
            selected_idx = layer_names.index(selected_layer_name)
            selected_layer = st.session_state.custom_layers[selected_idx]['layer']
            
            # Display the selected layer
            selected_layer_rgb = cv2.cvtColor(selected_layer, cv2.COLOR_BGR2RGB)
            st.image(selected_layer_rgb, caption=selected_layer_name, use_column_width=True)
            
            # Add download button for this layer
            layer_rgb_pil = Image.fromarray(selected_layer_rgb)
            layer_bytes = io.BytesIO()
            layer_rgb_pil.save(layer_bytes, format="PNG")
            
            st.download_button(
                label="Download This Layer",
                data=layer_bytes.getvalue(),
                file_name=f"{selected_layer_name.replace(' ', '_')}.png",
                mime="image/png",
                key=f"download_custom_{selected_idx}"
            )
            
            # Option to create black and white mask
            if st.button("Create B&W Mask"):
                # Create mask (white foreground, black background)
                mask = np.zeros((selected_layer.shape[0], selected_layer.shape[1]), dtype=np.uint8)
                is_fg = np.logical_not(np.all(selected_layer == bg_color_rgb, axis=2))
                mask[is_fg] = 255
                
                # Display the mask
                st.image(mask, caption=f"Mask for {selected_layer_name}", use_column_width=True)
                
                # Add download button for mask
                mask_pil = Image.fromarray(mask)
                mask_bytes = io.BytesIO()
                mask_pil.save(mask_bytes, format="PNG")
                
                st.download_button(
                    label="Download This Mask",
                    data=mask_bytes.getvalue(),
                    file_name=f"mask_{selected_layer_name.replace(' ', '_')}.png",
                    mime="image/png",
                    key=f"download_custom_mask_{selected_idx}"
                )

else:
    # Display sample usage when no image is uploaded
    st.markdown("<h2 class='sub-header'>How to use ColorSep</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    1. Upload an image using the sidebar file uploader
    2. Choose a color separation method:
       - **Exact color extraction**: Creates one layer per unique color, preserving all details
       - **K-means clustering**: Segments the image into distinct color clusters
       - **Dominant color extraction**: Extracts the most common colors
       - **Color thresholding**: Uses thresholds to separate colors
       - **LAB color space**: Uses perceptual color differences for more accurate separation
    3. Adjust parameters to fine-tune the separation
    4. View each color layer and combined preview
    5. Download individual layers or all layers as a zip file
    
    This tool is ideal for textile printing where each color needs to be printed separately.
    
    ### Advanced Features:
    - **Combine Layers**: Merge two color layers into a single layer
    - **Change Layer Colors**: Modify the color of any layer using RGB, Hex, or Pantone color codes
    """)
    
    st.info("⬅️ Use the sidebar to upload your image and get started!")
