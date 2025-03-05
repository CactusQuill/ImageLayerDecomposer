import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from skimage import color, segmentation, morphology, filters
import matplotlib.pyplot as plt
from scipy import ndimage

def apply_post_processing(mask, noise_reduction=0, apply_smoothing=False, smoothing_amount=0,
                         apply_sharpening=False, sharpening_amount=0):
    """Apply post-processing to a binary mask."""
    if noise_reduction > 0:
        # Remove small objects
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=noise_reduction * 10)
        mask = morphology.remove_small_holes(mask, area_threshold=noise_reduction * 10)
        
        # Apply morphological operations
        kernel = np.ones((noise_reduction, noise_reduction), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    if apply_smoothing and smoothing_amount > 0:
        # Apply Gaussian blur
        mask = cv2.GaussianBlur(mask.astype(np.float32), 
                               (smoothing_amount*2+1, smoothing_amount*2+1), 0)
    
    if apply_sharpening and sharpening_amount > 0:
        # Apply unsharp mask
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        mask = cv2.addWeighted(mask.astype(np.float32), 1.0 + sharpening_amount, 
                              blurred, -sharpening_amount, 0)
    
    # Ensure the mask is properly scaled
    mask = np.clip(mask, 0, 1).astype(np.uint8)
    
    return mask

def create_color_layer(img, mask, color, bg_color=(255, 255, 255)):
    """Create a color layer with the specified color for non-zero mask pixels."""
    h, w = mask.shape[:2]
    layer = np.full((h, w, 3), bg_color, dtype=np.uint8)  # Initialize with background color
    
    # Create a 3-channel mask
    mask_3ch = cv2.merge([mask, mask, mask])
    
    # Create a colored foreground
    colored_fg = np.full_like(layer, color)
    
    # Combine background and foreground based on mask
    # Where mask is non-zero, take the foreground color
    # Where mask is zero, keep the background
    layer = np.where(mask_3ch > 0, colored_fg, layer)
    
    return layer

def kmeans_color_separation(img, n_colors=5, compactness=1.0, bg_color=(255, 255, 255), 
                           noise_reduction=0, apply_smoothing=False, smoothing_amount=0,
                           apply_sharpening=False, sharpening_amount=0):
    """
    Separate an image into color layers using K-means clustering.
    
    Args:
        img: OpenCV image in BGR format
        n_colors: Number of color clusters to extract
        compactness: Controls the compactness of the clusters (higher = more compact)
        bg_color: Background color for the output layers (BGR)
        noise_reduction: Amount of noise reduction to apply (0 = none)
        apply_smoothing: Whether to apply smoothing to masks
        smoothing_amount: Amount of smoothing to apply
        apply_sharpening: Whether to apply sharpening to masks
        sharpening_amount: Amount of sharpening to apply
        
    Returns:
        Tuple of (color_layers, color_info)
    """
    # Reshape the image for K-means
    h, w = img.shape[:2]
    img_reshaped = img.reshape(-1, 3)
    
    # Apply K-means clustering with specified compactness
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(img_reshaped)
    
    # Get cluster centers (colors) and labels
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    # Calculate percentage of each color
    counts = Counter(labels)
    total_pixels = h * w
    percentages = {label: (count / total_pixels) * 100 for label, count in counts.items()}
    
    # Sort colors by occurrence (most common first)
    sorted_colors = sorted([(label, centers[label], percentages[label]) 
                           for label in counts], 
                          key=lambda x: x[2], reverse=True)
    
    # Create color layers
    color_layers = []
    color_info = []
    
    for label, color, percentage in sorted_colors:
        # Create binary mask for this color
        mask = np.zeros((h, w), dtype=np.uint8)
        mask_flat = np.zeros(total_pixels, dtype=np.uint8)
        mask_flat[labels == label] = 255
        mask = mask_flat.reshape(h, w)
        
        # Apply post-processing
        mask = apply_post_processing(
            mask, 
            noise_reduction=noise_reduction,
            apply_smoothing=apply_smoothing,
            smoothing_amount=smoothing_amount,
            apply_sharpening=apply_sharpening,
            sharpening_amount=sharpening_amount
        )
        
        # Create color layer
        layer = create_color_layer(img, mask, tuple(color), bg_color)
        
        color_layers.append(layer)
        color_info.append({'color': tuple(color), 'percentage': percentage})
    
    return color_layers, color_info

def dominant_color_separation(img, n_colors=5, min_percentage=1.0, bg_color=(255, 255, 255),
                             noise_reduction=0, apply_smoothing=False, smoothing_amount=0,
                             apply_sharpening=False, sharpening_amount=0):
    """
    Separate an image by extracting dominant colors and their regions.
    
    Args:
        img: OpenCV image in BGR format
        n_colors: Maximum number of colors to extract
        min_percentage: Minimum percentage of image coverage to include a color
        bg_color: Background color for the output layers
        noise_reduction: Amount of noise reduction to apply (0 = none)
        apply_smoothing: Whether to apply smoothing to masks
        smoothing_amount: Amount of smoothing to apply
        apply_sharpening: Whether to apply sharpening to masks
        sharpening_amount: Amount of sharpening to apply
        
    Returns:
        Tuple of (color_layers, color_info)
    """
    # Apply median blur to reduce noise while preserving edges
    img_blur = cv2.medianBlur(img, 5)
    
    # Convert to LAB color space for better perceptual distance
    img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
    
    # Reshape the image for color quantization
    h, w = img.shape[:2]
    pixels = img_lab.reshape(-1, 3)
    
    # Use K-means clustering with higher K to find more colors initially
    kmeans = KMeans(n_clusters=min(n_colors * 2, 20), random_state=42)
    kmeans.fit(pixels)
    
    # Get cluster centers and labels
    centers_lab = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    # Convert centers back to BGR
    centers = np.zeros_like(centers_lab)
    for i, center in enumerate(centers_lab):
        center_bgr = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_LAB2BGR)[0][0]
        centers[i] = center_bgr
    
    # Calculate percentage of each color
    counts = Counter(labels)
    total_pixels = h * w
    
    # Sort colors by occurrence and filter by minimum percentage
    color_data = []
    for label, count in counts.items():
        percentage = (count / total_pixels) * 100
        if percentage >= min_percentage:
            color_data.append((label, centers[label], percentage))
    
    # Sort by percentage and limit to n_colors
    color_data.sort(key=lambda x: x[2], reverse=True)
    color_data = color_data[:n_colors]
    
    # Create color layers
    color_layers = []
    color_info = []
    
    for label, color, percentage in color_data:
        # Create binary mask for this color
        mask = np.zeros((h, w), dtype=np.uint8)
        mask_flat = np.zeros(total_pixels, dtype=np.uint8)
        mask_flat[labels == label] = 255
        mask = mask_flat.reshape(h, w)
        
        # Apply post-processing
        mask = apply_post_processing(
            mask, 
            noise_reduction=noise_reduction,
            apply_smoothing=apply_smoothing,
            smoothing_amount=smoothing_amount,
            apply_sharpening=apply_sharpening,
            sharpening_amount=sharpening_amount
        )
        
        # Create color layer
        layer = create_color_layer(img, mask, tuple(color), bg_color)
        
        color_layers.append(layer)
        color_info.append({'color': tuple(color), 'percentage': percentage})
    
    return color_layers, color_info

def threshold_color_separation(img, threshold=25, blur_amount=3, bg_color=(255, 255, 255),
                              noise_reduction=0, apply_smoothing=False, smoothing_amount=0,
                              apply_sharpening=False, sharpening_amount=0):
    """
    Separate an image by color thresholding in multiple color spaces.
    
    Args:
        img: OpenCV image in BGR format
        threshold: Threshold value for color similarity
        blur_amount: Amount of blur to apply before thresholding
        bg_color: Background color for the output layers
        noise_reduction: Amount of noise reduction to apply (0 = none)
        apply_smoothing: Whether to apply smoothing to masks
        smoothing_amount: Amount of smoothing to apply
        apply_sharpening: Whether to apply sharpening to masks
        sharpening_amount: Amount of sharpening to apply
        
    Returns:
        Tuple of (color_layers, color_info)
    """
    h, w = img.shape[:2]
    
    # Apply blur to reduce noise
    if blur_amount > 0:
        img_blur = cv2.GaussianBlur(img, (blur_amount*2+1, blur_amount*2+1), 0)
    else:
        img_blur = img.copy()
    
    # Convert to different color spaces
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
    
    # Apply K-means to get initial color clusters
    kmeans = KMeans(n_clusters=min(10, threshold), random_state=42)
    kmeans.fit(img_blur.reshape(-1, 3))
    centers = kmeans.cluster_centers_.astype(int)
    
    # Create masks for each color cluster
    color_layers = []
    color_info = []
    
    for i, color in enumerate(centers):
        # Create masks in different color spaces
        # BGR mask (Euclidean distance)
        mask_bgr = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                # Calculate distance between pixel color and the current color
                dist = np.sqrt(np.sum((img_blur[y, x] - color)**2))
                if dist < threshold:
                    mask_bgr[y, x] = 255
        
        # Use connected components to get the largest regions
        num_labels, labels = cv2.connectedComponents(mask_bgr)
        if num_labels > 1:  # If there are connected components
            # Get the sizes of all connected components (excluding background)
            sizes = [np.sum(labels == i) for i in range(1, num_labels)]
            
            # Keep only the largest component
            largest_component = np.argmax(sizes) + 1
            mask_bgr = np.uint8(labels == largest_component) * 255
        
        # Apply post-processing
        mask = apply_post_processing(
            mask_bgr, 
            noise_reduction=noise_reduction,
            apply_smoothing=apply_smoothing,
            smoothing_amount=smoothing_amount,
            apply_sharpening=apply_sharpening,
            sharpening_amount=sharpening_amount
        )
        
        # Skip if mask is empty or too small
        if np.sum(mask) / 255 < (h * w * 0.01):  # Less than 1% of image
            continue
        
        # Create color layer
        layer = create_color_layer(img, mask, tuple(color), bg_color)
        
        # Calculate percentage
        percentage = (np.sum(mask) / 255 / (h * w)) * 100
        
        color_layers.append(layer)
        color_info.append({'color': tuple(color), 'percentage': percentage})
    
    # Sort by percentage (largest first)
    color_layers_sorted = []
    color_info_sorted = []
    sorted_indices = sorted(range(len(color_info)), 
                           key=lambda i: color_info[i]['percentage'], 
                           reverse=True)
    
    for idx in sorted_indices:
        color_layers_sorted.append(color_layers[idx])
        color_info_sorted.append(color_info[idx])
    
    return color_layers_sorted, color_info_sorted

def lab_color_separation(img, n_colors=5, delta_e=15, bg_color=(255, 255, 255),
                        noise_reduction=0, apply_smoothing=False, smoothing_amount=0,
                        apply_sharpening=False, sharpening_amount=0):
    """
    Separate an image into color layers using LAB color space and CIEDE2000 color difference.
    
    Args:
        img: OpenCV image in BGR format
        n_colors: Number of color clusters to extract
        delta_e: Delta E threshold for color similarity
        bg_color: Background color for the output layers (BGR)
        noise_reduction: Amount of noise reduction to apply (0 = none)
        apply_smoothing: Whether to apply smoothing to masks
        smoothing_amount: Amount of smoothing to apply
        apply_sharpening: Whether to apply sharpening to masks
        sharpening_amount: Amount of sharpening to apply
        
    Returns:
        Tuple of (color_layers, color_info)
    """
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Apply K-means clustering
    h, w = img.shape[:2]
    img_lab_reshaped = img_lab.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(img_lab_reshaped)
    
    # Get cluster centers and labels
    centers_lab = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Convert centers to BGR for display
    centers_bgr = []
    for center in centers_lab:
        # Reshape to format expected by cv2.cvtColor
        center_lab = np.uint8([[center]])
        center_bgr = cv2.cvtColor(center_lab, cv2.COLOR_LAB2BGR)[0][0]
        centers_bgr.append(center_bgr)
    
    # Calculate percentage of each color
    counts = Counter(labels)
    total_pixels = h * w
    percentages = {label: (count / total_pixels) * 100 for label, count in counts.items()}
    
    # Sort colors by occurrence (most common first)
    sorted_colors = sorted([(label, centers_bgr[label], percentages[label]) 
                           for label in counts], 
                          key=lambda x: x[2], reverse=True)
    
    # Create color layers using Delta E in LAB space
    color_layers = []
    color_info = []
    
    for label, color_bgr, percentage in sorted_colors:
        # Get LAB color center
        color_lab = centers_lab[label]
        
        # Create mask based on Delta E
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Compute Delta E for each pixel
        for y in range(h):
            for x in range(w):
                pixel_lab = img_lab[y, x]
                
                # Calculate simple Euclidean distance in LAB space
                # (approximation of Delta E)
                delta = np.sqrt(np.sum((pixel_lab - color_lab) ** 2))
                
                if delta < delta_e:
                    mask[y, x] = 255
        
        # Apply post-processing
        mask = apply_post_processing(
            mask, 
            noise_reduction=noise_reduction,
            apply_smoothing=apply_smoothing,
            smoothing_amount=smoothing_amount,
            apply_sharpening=apply_sharpening,
            sharpening_amount=sharpening_amount
        )
        
        # Create color layer
        layer = create_color_layer(img, mask, tuple(color_bgr.astype(int)), bg_color)
        
        color_layers.append(layer)
        color_info.append({'color': tuple(color_bgr.astype(int)), 'percentage': percentage})
    
    return color_layers, color_info

def exact_color_separation(img, max_colors=100, bg_color=(255, 255, 255)):
    """
    Separate an image by extracting EXACT colors without any clustering or approximation.
    This method preserves all details and creates a separate layer for each unique color.
    
    Args:
        img: OpenCV image in BGR format
        max_colors: Maximum number of colors to extract (to avoid too many layers)
        bg_color: Background color for the output layers (BGR)
        
    Returns:
        Tuple of (color_layers, color_info)
    """
    h, w = img.shape[:2]
    
    # Get unique colors from the image with their counts
    # Reshape the image to a 2D array of pixels
    pixels = img.reshape(-1, 3)
    
    # Find unique colors and their counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    # Convert to a list of tuples for sorting
    color_counts = [(tuple(color), count) for color, count in zip(unique_colors, counts)]
    
    # Sort by count (most common first)
    color_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Limit to max_colors
    if len(color_counts) > max_colors:
        print(f"Image has {len(color_counts)} unique colors. Limiting to top {max_colors}.")
        color_counts = color_counts[:max_colors]
    
    # Calculate total pixels for percentage
    total_pixels = h * w
    
    # Create color layers
    color_layers = []
    color_info = []
    
    for color, count in color_counts:
        # Skip colors that are too rare (less than 0.01% of the image)
        if count / total_pixels < 0.0001:
            continue
            
        # Create a binary mask for this exact color
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # This is a vectorized approach that is much faster than pixel-by-pixel
        # Create a boolean mask where all three channels match the current color
        r_match = img[:,:,0] == color[0]
        g_match = img[:,:,1] == color[1]
        b_match = img[:,:,2] == color[2]
        
        # Combine the channel matches
        color_match = r_match & g_match & b_match
        
        # Set matching pixels to 255 in the mask
        mask[color_match] = 255
        
        # Create color layer
        layer = create_color_layer(img, mask, color, bg_color)
        
        # Calculate percentage
        percentage = (count / total_pixels) * 100
        
        color_layers.append(layer)
        color_info.append({'color': color, 'percentage': percentage})
    
    return color_layers, color_info

def combine_layers(layer1, layer2, color=None, bg_color=(255, 255, 255)):
    """
    Combine two color layers into a new layer.
    
    Args:
        layer1: First layer (BGR format)
        layer2: Second layer (BGR format)
        color: Optional color to use for the combined layer (BGR format)
               If None, keep the original colors from both layers
        bg_color: Background color for the output layer (BGR)
        
    Returns:
        Combined layer
    """
    # Ensure both layers have the same dimensions
    if layer1.shape != layer2.shape:
        raise ValueError("Layers must have the same dimensions")
    
    h, w = layer1.shape[:2]
    
    # Create masks for non-background pixels in each layer
    # Background pixels are those matching bg_color exactly
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    
    # Create boolean masks for non-background pixels
    is_fg1 = np.logical_not(np.all(layer1 == bg_color, axis=2))
    is_fg2 = np.logical_not(np.all(layer2 == bg_color, axis=2))
    
    # Set mask values
    mask1[is_fg1] = 255
    mask2[is_fg2] = 255
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask1, mask2)
    
    # Create combined layer
    if color is not None:
        # Use specified color for all foreground pixels
        combined_layer = create_color_layer(np.zeros_like(layer1), combined_mask, color, bg_color)
    else:
        # Initialize with background color
        combined_layer = np.full((h, w, 3), bg_color, dtype=np.uint8)
        
        # First add layer1 content
        combined_layer[is_fg1] = layer1[is_fg1]
        
        # Then add layer2 content (overwriting layer1 where they overlap)
        combined_layer[is_fg2] = layer2[is_fg2]
    
    return combined_layer

def change_layer_color(layer, new_color, bg_color=(255, 255, 255)):
    """
    Change the color of a layer while preserving its shape/mask.
    
    Args:
        layer: Layer to modify (BGR format)
        new_color: New color to apply (BGR format)
        bg_color: Background color of the layer (BGR)
        
    Returns:
        Recolored layer
    """
    h, w = layer.shape[:2]
    
    # Create mask for non-background pixels
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Background pixels are those matching bg_color exactly
    is_bg = np.all(layer == bg_color, axis=2)
    is_fg = np.logical_not(is_bg)
    
    # Set mask values
    mask[is_fg] = 255
    
    # Create new layer with the specified color
    new_layer = create_color_layer(np.zeros_like(layer), mask, new_color, bg_color)
    
    return new_layer

# Additional layer manipulation functions

def invert_layer(layer, bg_color=(255, 255, 255)):
    """
    Invert a layer's mask while preserving its color.
    
    Args:
        layer: Layer to invert (BGR format)
        bg_color: Background color of the layer (BGR)
        
    Returns:
        Inverted layer
    """
    # Extract the mask by checking which pixels are not background color
    mask = np.any(layer != bg_color, axis=2).astype(np.uint8) * 255
    
    # Invert the mask
    inverted_mask = cv2.bitwise_not(mask)
    
    # Get the original color (assuming uniform color)
    non_bg_pixels = layer[layer != bg_color.reshape(1, 1, 3)]
    if len(non_bg_pixels) > 0:
        color = non_bg_pixels.reshape(-1, 3)[0]
    else:
        # If no foreground pixels, use a default color
        color = (0, 0, 0)
    
    # Create a new layer with the inverted mask and original color
    inverted_layer = create_color_layer(layer, inverted_mask, color, bg_color)
    
    return inverted_layer

def erode_dilate_layer(layer, operation='erode', kernel_size=3, iterations=1, bg_color=(255, 255, 255)):
    """
    Apply erosion or dilation to a layer's mask.
    
    Args:
        layer: Layer to modify (BGR format)
        operation: 'erode' or 'dilate'
        kernel_size: Size of the kernel for morphological operation
        iterations: Number of iterations to apply the operation
        bg_color: Background color of the layer (BGR)
        
    Returns:
        Modified layer
    """
    # Extract the mask
    mask = np.any(layer != bg_color, axis=2).astype(np.uint8) * 255
    
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply operation
    if operation == 'erode':
        new_mask = cv2.erode(mask, kernel, iterations=iterations)
    elif operation == 'dilate':
        new_mask = cv2.dilate(mask, kernel, iterations=iterations)
    else:
        return layer  # Return original if invalid operation
    
    # Get the original color
    non_bg_pixels = layer[layer != bg_color.reshape(1, 1, 3)]
    if len(non_bg_pixels) > 0:
        color = non_bg_pixels.reshape(-1, 3)[0]
    else:
        color = (0, 0, 0)
    
    # Create a new layer with the modified mask
    modified_layer = create_color_layer(layer, new_mask, color, bg_color)
    
    return modified_layer

def transform_layer(layer, operation='rotate90', bg_color=(255, 255, 255)):
    """
    Apply geometric transformations to a layer.
    
    Args:
        layer: Layer to transform (BGR format)
        operation: One of 'rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v'
        bg_color: Background color of the layer (BGR)
        
    Returns:
        Transformed layer
    """
    if operation == 'rotate90':
        transformed = cv2.rotate(layer, cv2.ROTATE_90_CLOCKWISE)
    elif operation == 'rotate180':
        transformed = cv2.rotate(layer, cv2.ROTATE_180)
    elif operation == 'rotate270':
        transformed = cv2.rotate(layer, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif operation == 'flip_h':
        transformed = cv2.flip(layer, 1)  # 1 = horizontal flip
    elif operation == 'flip_v':
        transformed = cv2.flip(layer, 0)  # 0 = vertical flip
    else:
        return layer  # Return original if invalid operation
    
    return transformed

def adjust_layer_opacity(layer, opacity=0.5, bg_color=(255, 255, 255)):
    """
    Adjust the opacity of a layer.
    
    Args:
        layer: Layer to modify (BGR format)
        opacity: Opacity level from 0.0 to 1.0
        bg_color: Background color of the layer (BGR)
        
    Returns:
        Layer with adjusted opacity
    """
    # Create a copy of the layer
    result = layer.copy()
    
    # Extract the mask (where pixels are not background)
    mask = np.any(layer != bg_color, axis=2)
    
    # For each foreground pixel, blend with background based on opacity
    for i in range(3):  # For each channel
        result[:,:,i] = np.where(
            mask,
            (layer[:,:,i] * opacity + bg_color[i] * (1 - opacity)).astype(np.uint8),
            bg_color[i]
        )
    
    return result

def apply_blur_sharpen(layer, operation='blur', amount=5, bg_color=(255, 255, 255)):
    """
    Apply blur or sharpen filter to a layer.
    
    Args:
        layer: Layer to modify (BGR format)
        operation: 'blur' or 'sharpen'
        amount: Intensity of the effect
        bg_color: Background color of the layer (BGR)
        
    Returns:
        Modified layer
    """
    # Extract the mask
    mask = np.any(layer != bg_color, axis=2).astype(np.uint8) * 255
    
    # Get the foreground color
    non_bg_pixels = layer[layer != bg_color.reshape(1, 1, 3)]
    if len(non_bg_pixels) > 0:
        color = non_bg_pixels.reshape(-1, 3)[0]
    else:
        color = (0, 0, 0)
    
    # Apply the selected operation to the mask
    if operation == 'blur':
        # Apply Gaussian blur
        modified_mask = cv2.GaussianBlur(mask, (amount*2+1, amount*2+1), 0)
    elif operation == 'sharpen':
        # Apply unsharp mask technique for sharpening
        gaussian = cv2.GaussianBlur(mask, (5, 5), 0)
        modified_mask = cv2.addWeighted(mask, 1.0 + amount/10.0, gaussian, -amount/10.0, 0)
        modified_mask = np.clip(modified_mask, 0, 255).astype(np.uint8)
    else:
        return layer  # Return original if invalid operation
    
    # Create a new layer with the modified mask
    modified_layer = create_color_layer(layer, modified_mask, color, bg_color)
    
    return modified_layer

def apply_threshold(layer, threshold_value=127, bg_color=(255, 255, 255)):
    """
    Apply threshold to a layer to make mask more binary.
    
    Args:
        layer: Layer to modify (BGR format)
        threshold_value: Threshold value (0-255)
        bg_color: Background color of the layer (BGR)
        
    Returns:
        Modified layer with thresholded mask
    """
    # Extract the mask
    mask = np.any(layer != bg_color, axis=2).astype(np.uint8) * 255
    
    # Apply threshold
    _, thresholded_mask = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Get the foreground color
    non_bg_pixels = layer[layer != bg_color.reshape(1, 1, 3)]
    if len(non_bg_pixels) > 0:
        color = non_bg_pixels.reshape(-1, 3)[0]
    else:
        color = (0, 0, 0)
    
    # Create a new layer with the thresholded mask
    modified_layer = create_color_layer(layer, thresholded_mask, color, bg_color)
    
    return modified_layer

# Pantone TPX and TPG color mappings (limited set, can be expanded)
# Format: 'CODE': (R, G, B)
PANTONE_TPX = {
    '19-4052 TCX': (15, 76, 129),    # Classic Blue (Color of the Year 2020)
    '16-1546 TCX': (244, 76, 113),   # Living Coral (Color of the Year 2019)
    '18-3838 TCX': (101, 78, 163),   # Ultra Violet (Color of the Year 2018)
    '15-0343 TCX': (136, 176, 75),   # Greenery (Color of the Year 2017)
    '13-1520 TCX': (247, 202, 201),  # Rose Quartz (Color of the Year 2016)
    '14-4313 TCX': (145, 168, 208),  # Serenity (Color of the Year 2016)
    '18-1438 TCX': (141, 58, 41),    # Marsala (Color of the Year 2015)
    '17-1360 TCX': (186, 59, 30),    # Tangerine Tango (Color of the Year 2012)
    '11-0601 TCX': (237, 234, 218),  # Whisper White
    '19-4005 TCX': (44, 44, 44),     # Black
    '19-1664 TCX': (163, 0, 43),     # True Red
    '17-1462 TCX': (221, 65, 36),    # Flame Orange
    '14-0756 TCX': (212, 169, 0),    # Yellow Gold
    '15-5534 TCX': (0, 151, 132),    # Turquoise
    '19-3950 TCX': (72, 40, 125),    # Purple
    '18-0135 TCX': (0, 139, 54),     # Kelly Green
    '14-4122 TCX': (85, 180, 233),   # Sky Blue
}

PANTONE_TPG = {
    '19-4052 TPG': (40, 96, 143),    # Classic Blue (Color of the Year 2020)
    '16-1546 TPG': (233, 81, 96),    # Living Coral (Color of the Year 2019)
    '18-3838 TPG': (117, 83, 145),   # Ultra Violet (Color of the Year 2018)
    '15-0343 TPG': (136, 171, 81),   # Greenery (Color of the Year 2017)
    '13-1520 TPG': (242, 190, 182),  # Rose Quartz (Color of the Year 2016)
    '14-4313 TPG': (146, 170, 199),  # Serenity (Color of the Year 2016)
    '18-1438 TPG': (149, 78, 55),    # Marsala (Color of the Year 2015)
    '17-1360 TPG': (191, 76, 47),    # Tangerine Tango (Color of the Year 2012)
    '11-0601 TPG': (232, 228, 213),  # Whisper White
    '19-4005 TPG': (60, 60, 59),     # Black
    '19-1664 TPG': (168, 26, 49),    # True Red
    '17-1462 TPG': (218, 83, 44),    # Flame Orange
    '14-0756 TPG': (214, 170, 40),   # Yellow Gold
    '15-5534 TPG': (0, 148, 126),    # Turquoise
    '19-3950 TPG': (82, 55, 113),    # Purple
    '18-0135 TPG': (0, 131, 62),     # Kelly Green
    '14-4122 TPG': (94, 175, 221),   # Sky Blue
}

def get_color_from_code(color_code):
    """
    Convert a color code to BGR color value.
    
    Args:
        color_code: Color code string (RGB or Pantone)
        
    Returns:
        BGR color tuple
    """
    # Handle RGB format (r,g,b)
    if color_code.startswith('(') and color_code.endswith(')'):
        try:
            # Parse RGB values
            rgb = eval(color_code)
            if isinstance(rgb, tuple) and len(rgb) == 3:
                # Convert RGB to BGR
                return (rgb[2], rgb[1], rgb[0])
        except:
            pass
    
    # Handle hex format (#rrggbb)
    if color_code.startswith('#') and len(color_code) == 7:
        try:
            r = int(color_code[1:3], 16)
            g = int(color_code[3:5], 16)
            b = int(color_code[5:7], 16)
            return (b, g, r)  # BGR format for OpenCV
        except:
            pass
    
    # Handle Pantone TPX format
    if 'TPX' in color_code and color_code in PANTONE_TPX:
        rgb = PANTONE_TPX[color_code]
        return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR
    
    # Handle Pantone TPG format
    if 'TPG' in color_code and color_code in PANTONE_TPG:
        rgb = PANTONE_TPG[color_code]
        return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR
    
    # Return black if format not recognized
    return (0, 0, 0)
