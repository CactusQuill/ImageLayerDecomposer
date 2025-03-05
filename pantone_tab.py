"""
This module contains the Pantone Color Extraction tab functionality
for the ColorSep application.
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import zipfile
import os
import io
from io import BytesIO
from PIL import Image
from datetime import datetime
from sklearn.cluster import KMeans
import pantone_colors as pantone

def pantone_extraction_tab():
    """Renders the Pantone Color Extraction tab content."""
    
    st.header("Pantone Color Extraction")
    st.write("Extract layers based on specific Pantone TPG/TPX colors.")

    # Upload section for Pantone extraction
    uploaded_file_pantone = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"], key="pantone_uploader")
    
    if uploaded_file_pantone is not None:
        # Process the image
        image_bytes = uploaded_file_pantone.getvalue()
        pantone_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        pantone_rgb_image = cv2.cvtColor(pantone_image, cv2.COLOR_BGR2RGB)
        
        # Display the original image
        st.image(pantone_rgb_image, caption="Original Image", use_column_width=True)
        
        # Color selection interface
        st.subheader("Color Selection")
        
        # Two methods to specify colors:
        # 1. Select from predefined Pantone colors
        # 2. Extract from image and find closest Pantone match
        color_selection_method = st.radio(
            "Color Selection Method",
            ["Select Pantone Colors", "Extract from Image"],
            horizontal=True
        )
        
        pantone_colors_selected = []
        
        if color_selection_method == "Select Pantone Colors":
            # Display a multi-select for Pantone colors with color swatches
            st.write("Select Pantone colors to extract from the image:")
            
            # Create columns to show color swatches
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Show Golden Yellow
                st.markdown(f'<div style="background-color: rgb{pantone.PANTONE_TO_RGB["14-0952 TPG"]}; width: 50px; height: 50px; display: inline-block; margin-right: 10px;"></div> 14-0952 TPG (Golden Yellow)', unsafe_allow_html=True)
                if st.checkbox("14-0952 TPG"):
                    pantone_colors_selected.append("14-0952 TPG")
            
            with col2:
                # Show Strong Blue
                st.markdown(f'<div style="background-color: rgb{pantone.PANTONE_TO_RGB["18-4051 TPG"]}; width: 50px; height: 50px; display: inline-block; margin-right: 10px;"></div> 18-4051 TPG (Strong Blue)', unsafe_allow_html=True)
                if st.checkbox("18-4051 TPG"):
                    pantone_colors_selected.append("18-4051 TPG")
            
            with col3:
                # Show White
                st.markdown(f'<div style="background-color: rgb{pantone.PANTONE_TO_RGB["WHITE"]}; width: 50px; height: 50px; display: inline-block; margin-right: 10px; border: 1px solid #ddd;"></div> WHITE (Pure White)', unsafe_allow_html=True)
                if st.checkbox("WHITE"):
                    pantone_colors_selected.append("WHITE")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Show Soft Coral
                st.markdown(f'<div style="background-color: rgb{pantone.PANTONE_TO_RGB["15-1523 TPG"]}; width: 50px; height: 50px; display: inline-block; margin-right: 10px;"></div> 15-1523 TPG (Soft Coral)', unsafe_allow_html=True)
                if st.checkbox("15-1523 TPG"):
                    pantone_colors_selected.append("15-1523 TPG")
            
            with col2:
                # Show Brown
                st.markdown(f'<div style="background-color: rgb{pantone.PANTONE_TO_RGB["19-1334 TPX"]}; width: 50px; height: 50px; display: inline-block; margin-right: 10px;"></div> 19-1334 TPX (Brown)', unsafe_allow_html=True)
                if st.checkbox("19-1334 TPX"):
                    pantone_colors_selected.append("19-1334 TPX")
            
            with col3:
                # Show Light Beige
                st.markdown(f'<div style="background-color: rgb{pantone.PANTONE_TO_RGB["14-1116 TPX"]}; width: 50px; height: 50px; display: inline-block; margin-right: 10px;"></div> 14-1116 TPX (Light Beige)', unsafe_allow_html=True)
                if st.checkbox("14-1116 TPX"):
                    pantone_colors_selected.append("14-1116 TPX")
            
            # Option to add more Pantone colors
            with st.expander("Add More Pantone Colors"):
                # Create a searchable dropdown for Pantone codes
                all_pantone_codes = list(pantone.PANTONE_TO_RGB.keys())
                selected_code = st.selectbox("Select a Pantone code", all_pantone_codes)
                
                # Show a color preview
                if selected_code:
                    rgb_color = pantone.PANTONE_TO_RGB[selected_code]
                    st.markdown(f'<div style="background-color: rgb{rgb_color}; width: 100px; height: 50px; display: inline-block; margin-right: 10px;"></div> {selected_code}', unsafe_allow_html=True)
                    
                    if st.button(f"Add {selected_code}"):
                        if selected_code not in pantone_colors_selected:
                            pantone_colors_selected.append(selected_code)
                            st.success(f"Added {selected_code}")
            
        else:  # Extract from Image
            st.write("Click on the image to select colors to extract, or set the number of colors to auto-extract:")
            
            # Option to auto-extract dominant colors
            num_colors = st.slider("Number of colors to extract", 1, 10, 6)
            
            if st.button("Auto-Extract Colors"):
                # Resize image for faster processing if it's too large
                h, w = pantone_image.shape[:2]
                max_dim = 400
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    pantone_image_small = cv2.resize(pantone_image, (new_w, new_h))
                else:
                    pantone_image_small = pantone_image.copy()
                
                # Reshape for K-means
                pixels = pantone_image_small.reshape(-1, 3)
                
                # K-means clustering to find dominant colors
                kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Get the dominant colors
                dominant_colors = kmeans.cluster_centers_.astype(int)
                
                # Find closest Pantone colors for each dominant color
                st.subheader("Extracted Colors and Their Pantone Matches")
                
                # Create empty list to store selected Pantone colors
                extracted_pantone_colors = []
                
                for i, color in enumerate(dominant_colors):
                    # BGR to RGB for display and matching
                    rgb_color = (int(color[2]), int(color[1]), int(color[0]))
                    
                    # Find closest Pantone match using LAB color space
                    matches = pantone.find_closest_pantone_lab(rgb_color, tolerance=40)
                    
                    if matches:
                        pantone_code = matches[0][0]
                        pantone_rgb = matches[0][1]
                        color_distance = matches[0][2]
                        
                        # Show the extracted color and its Pantone match
                        col1, col2, col3 = st.columns([1, 1, 3])
                        
                        with col1:
                            st.markdown(f'<div style="background-color: rgb{rgb_color}; width: 50px; height: 50px; border: 1px solid #ddd;"></div>', unsafe_allow_html=True)
                            st.write(f"RGB: {rgb_color}")
                        
                        with col2:
                            st.markdown(f'<div style="background-color: rgb{pantone_rgb}; width: 50px; height: 50px; border: 1px solid #ddd;"></div>', unsafe_allow_html=True)
                            st.write(f"{pantone_code}")
                        
                        with col3:
                            st.write(f"Match confidence: {100 - min(100, color_distance * 3):.1f}%")
                            if st.checkbox(f"Use {pantone_code}", key=f"extract_{i}"):
                                extracted_pantone_colors.append(pantone_code)
                
                # Add the selected colors to the main selection list
                pantone_colors_selected.extend(extracted_pantone_colors)
        
        # Show the selected colors
        if pantone_colors_selected:
            st.subheader("Selected Pantone Colors")
            st.write(f"You have selected {len(pantone_colors_selected)} colors:")
            
            # Display the selected colors in a grid
            cols = st.columns(min(3, len(pantone_colors_selected)))
            for i, code in enumerate(pantone_colors_selected):
                with cols[i % 3]:
                    rgb = pantone.PANTONE_TO_RGB[code]
                    st.markdown(f'<div style="background-color: rgb{rgb}; width: 80px; height: 50px; border: 1px solid #ddd;"></div>', unsafe_allow_html=True)
                    st.write(f"{code}")
            
            # Parameters for extraction
            st.subheader("Extraction Parameters")
            color_tolerance = st.slider("Color Tolerance (higher = more inclusive)", 5, 50, 20)
            
            # Button to extract layers
            if st.button("Extract Pantone Color Layers"):
                st.subheader("Extracted Layers")
                
                # Create a list to store the extracted layers
                pantone_layers = []
                
                # Process each selected Pantone color
                for code in pantone_colors_selected:
                    # Get the RGB values for this Pantone color
                    rgb_color = pantone.PANTONE_TO_RGB[code]
                    
                    # Convert to BGR for OpenCV
                    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                    
                    # Create a mask for this color
                    # Calculate distance from each pixel to the target color
                    color_distances = np.sqrt(np.sum((pantone_image.astype(int) - bgr_color) ** 2, axis=2))
                    
                    # Create a mask where distance is less than the tolerance
                    color_mask = (color_distances <= color_tolerance).astype(np.uint8) * 255
                    
                    # Create a colored layer
                    layer = np.zeros_like(pantone_image)
                    layer[color_mask > 0] = bgr_color
                    
                    # Store the layer
                    pantone_layers.append((code, layer, color_mask))
                    
                    # Display the extracted layer
                    rgb_layer = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                    st.image(rgb_layer, caption=f"Layer for {code}", use_column_width=True)
                    
                    # Add download button for this layer
                    # Convert BGR to RGB for PIL
                    rgb_layer_pil = Image.fromarray(rgb_layer)
                    buffer = BytesIO()
                    rgb_layer_pil.save(buffer, format="PNG")
                    
                    st.download_button(
                        label=f"Download {code} Layer",
                        data=buffer.getvalue(),
                        file_name=f"pantone_{code.replace(' ', '_')}.png",
                        mime="image/png",
                    )
                    
                    # Add download button for mask layer
                    mask_pil = Image.fromarray(color_mask)
                    buffer = BytesIO()
                    mask_pil.save(buffer, format="PNG")
                    
                    st.download_button(
                        label=f"Download {code} Mask",
                        data=buffer.getvalue(),
                        file_name=f"pantone_mask_{code.replace(' ', '_')}.png",
                        mime="image/png",
                    )
                
                # Create a combined preview of all layers
                if pantone_layers:
                    st.subheader("Combined Preview")
                    combined = np.zeros_like(pantone_image)
                    
                    # Start with a white background
                    combined.fill(255)
                    
                    # Add each layer
                    for code, layer, mask in pantone_layers:
                        # Overlay on the white background
                        mask_3ch = np.stack([mask] * 3, axis=2) / 255.0
                        combined = (combined * (1 - mask_3ch) + layer * mask_3ch).astype(np.uint8)
                    
                    # Display the combined result
                    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                    st.image(combined_rgb, caption="Combined Pantone Layers", use_column_width=True)
                    
                    # Add download button for combined preview
                    combined_rgb_pil = Image.fromarray(combined_rgb)
                    buffer = BytesIO()
                    combined_rgb_pil.save(buffer, format="PNG")
                    
                    st.download_button(
                        label="Download Combined Preview",
                        data=buffer.getvalue(),
                        file_name=f"pantone_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                    )
                    
                    # Add download all layers button
                    with st.spinner("Preparing all layers for download..."):
                        # Create a temporary directory
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            # Save each layer
                            layer_files = []
                            for i, (code, layer, mask) in enumerate(pantone_layers):
                                # Save the color layer
                                layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)
                                layer_filename = f"pantone_{code.replace(' ', '_')}.png"
                                layer_path = os.path.join(tmpdirname, layer_filename)
                                Image.fromarray(layer_rgb).save(layer_path)
                                layer_files.append(layer_path)
                                
                                # Save the mask layer
                                mask_filename = f"pantone_mask_{code.replace(' ', '_')}.png"
                                mask_path = os.path.join(tmpdirname, mask_filename)
                                Image.fromarray(mask).save(mask_path)
                                layer_files.append(mask_path)
                            
                            # Save the combined preview
                            combined_filename = "pantone_combined.png"
                            combined_path = os.path.join(tmpdirname, combined_filename)
                            Image.fromarray(combined_rgb).save(combined_path)
                            layer_files.append(combined_path)
                            
                            # Create a README file
                            readme_filename = "README.txt"
                            readme_path = os.path.join(tmpdirname, readme_filename)
                            with open(readme_path, 'w') as readme_file:
                                readme_file.write("Pantone Color Extraction\n")
                                readme_file.write("========================\n\n")
                                readme_file.write(f"Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                                readme_file.write("Extracted Pantone Colors:\n")
                                for code in pantone_colors_selected:
                                    readme_file.write(f"- {code}\n")
                                readme_file.write("\nFile descriptions:\n")
                                readme_file.write("- pantone_[code].png: Full color layer for each Pantone code\n")
                                readme_file.write("- pantone_mask_[code].png: Black and white mask for each Pantone code\n")
                                readme_file.write("- pantone_combined.png: All layers combined into a single image\n")
                            
                            layer_files.append(readme_path)
                            
                            # Create a zip file containing all layers
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                                for file in layer_files:
                                    zip_file.write(file, os.path.basename(file))
                            
                            # Download button for zip file
                            st.download_button(
                                label="Download All Pantone Layers",
                                data=zip_buffer.getvalue(),
                                file_name=f"pantone_layers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                            )
