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
