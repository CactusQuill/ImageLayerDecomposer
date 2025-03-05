"""
Helper module with Pantone color codes and a function to get all available color codes for display in the UI.
"""

def get_all_pantone_codes():
    """
    Returns a dictionary of all available Pantone color codes with their names.
    """
    pantone_colors = {
        # TPX Colors (Textile Paper eXtended)
        '19-4052 TCX': 'Classic Blue (2020)',
        '16-1546 TCX': 'Living Coral (2019)',
        '18-3838 TCX': 'Ultra Violet (2018)',
        '15-0343 TCX': 'Greenery (2017)',
        '13-1520 TCX': 'Rose Quartz (2016)',
        '14-4313 TCX': 'Serenity (2016)',
        '18-1438 TCX': 'Marsala (2015)',
        '17-1360 TCX': 'Tangerine Tango (2012)',
        '11-0601 TCX': 'Whisper White',
        '19-4005 TCX': 'Black',
        '19-1664 TCX': 'True Red',
        '17-1462 TCX': 'Flame Orange',
        '14-0756 TCX': 'Yellow Gold',
        '15-5534 TCX': 'Turquoise',
        '19-3950 TCX': 'Purple',
        '18-0135 TCX': 'Kelly Green',
        '14-4122 TCX': 'Sky Blue',
        
        # TPG Colors (Textile Paper Gloss)
        '19-4052 TPG': 'Classic Blue (2020)',
        '16-1546 TPG': 'Living Coral (2019)',
        '18-3838 TPG': 'Ultra Violet (2018)',
        '15-0343 TPG': 'Greenery (2017)',
        '13-1520 TPG': 'Rose Quartz (2016)',
        '14-4313 TPG': 'Serenity (2016)',
        '18-1438 TPG': 'Marsala (2015)',
        '17-1360 TPG': 'Tangerine Tango (2012)',
        '11-0601 TPG': 'Whisper White',
        '19-4005 TPG': 'Black',
        '19-1664 TPG': 'True Red',
        '17-1462 TPG': 'Flame Orange',
        '14-0756 TPG': 'Yellow Gold',
        '15-5534 TPG': 'Turquoise',
        '19-3950 TPG': 'Purple',
        '18-0135 TPG': 'Kelly Green',
        '14-4122 TPG': 'Sky Blue',
        
        # Additional TPG Colors from the image example
        '14-0952 TPG': 'Golden Yellow',
        '18-4051 TPG': 'Strong Blue',
        'WHITE': 'Pure White',
        '15-1523 TPG': 'Soft Coral',
        '19-1334 TPX': 'Brown',
        '14-1116 TPX': 'Light Beige',
    }
    
    return pantone_colors


# RGB values for Pantone TPG and TPX colors
# These are approximate values based on common conversions
PANTONE_TO_RGB = {
    # TPG Colors
    '14-0952 TPG': (237, 176, 33),  # Golden Yellow
    '18-4051 TPG': (45, 96, 163),   # Strong Blue
    'WHITE': (255, 255, 255),       # Pure White
    '15-1523 TPG': (236, 141, 130), # Soft Coral
    '19-1334 TPX': (125, 65, 27),   # Brown
    '14-1116 TPX': (226, 198, 158), # Light Beige
    
    # Common TPG colors
    '19-4052 TPG': (15, 76, 129),    # Classic Blue
    '16-1546 TPG': (250, 114, 104),  # Living Coral
    '18-3838 TPG': (95, 75, 139),    # Ultra Violet
    '15-0343 TPG': (136, 176, 75),   # Greenery
    '13-1520 TPG': (242, 189, 205),  # Rose Quartz
    '14-4313 TPG': (145, 168, 208),  # Serenity
    '18-1438 TPG': (141, 60, 75),    # Marsala
    '17-1360 TPG': (221, 65, 36),    # Tangerine Tango
    '11-0601 TPG': (237, 241, 255),  # Whisper White
    '19-4005 TPG': (38, 38, 38),     # Black
    '19-1664 TPG': (186, 32, 38),    # True Red
    '17-1462 TPG': (237, 92, 40),    # Flame Orange
    '14-0756 TPG': (227, 183, 35),   # Yellow Gold
    '15-5534 TPG': (72, 176, 175),   # Turquoise
    '19-3950 TPG': (82, 35, 115),    # Purple
    '18-0135 TPG': (0, 127, 70),     # Kelly Green
    '14-4122 TPG': (108, 172, 212),  # Sky Blue
}


def get_rgb_from_pantone(pantone_code):
    """
    Returns the RGB values for a given Pantone color code.
    """
    if pantone_code in PANTONE_TO_RGB:
        return PANTONE_TO_RGB[pantone_code]
    
    # If not found, return None
    return None


def find_closest_pantone_to_rgb(target_rgb, tolerance=30):
    """
    Finds the closest Pantone color to the given RGB value.
    
    Args:
        target_rgb (tuple): The RGB value to match (R, G, B)
        tolerance (int): Maximum color distance to consider a match
        
    Returns:
        list: A list of tuples (pantone_code, rgb_value, distance) sorted by distance
    """
    import math
    
    matches = []
    
    for code, rgb in PANTONE_TO_RGB.items():
        # Calculate Euclidean distance between colors
        r_diff = target_rgb[0] - rgb[0]
        g_diff = target_rgb[1] - rgb[1]
        b_diff = target_rgb[2] - rgb[2]
        
        distance = math.sqrt(r_diff**2 + g_diff**2 + b_diff**2)
        
        if distance <= tolerance:
            matches.append((code, rgb, distance))
    
    # Sort by distance (closest first)
    return sorted(matches, key=lambda x: x[2])


def rgb_to_lab(rgb):
    """Convert RGB to LAB color space for more perceptually accurate color matching"""
    import numpy as np
    
    # Normalize RGB values
    r, g, b = [x/255 for x in rgb]
    
    # Convert to XYZ
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
    
    r *= 100
    g *= 100
    b *= 100
    
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    
    # Normalize using D65 white point
    x /= 95.047
    y /= 100.0
    z /= 108.883
    
    # Convert XYZ to LAB
    x = x ** (1/3) if x > 0.008856 else 7.787 * x + 16/116
    y = y ** (1/3) if y > 0.008856 else 7.787 * y + 16/116
    z = z ** (1/3) if z > 0.008856 else 7.787 * z + 16/116
    
    L = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    
    return (L, a, b)


def find_closest_pantone_lab(target_rgb, tolerance=25):
    """
    Finds the closest Pantone color to the given RGB value using LAB color space.
    
    Args:
        target_rgb (tuple): The RGB value to match (R, G, B)
        tolerance (float): Maximum color distance to consider a match
        
    Returns:
        list: A list of tuples (pantone_code, rgb_value, distance) sorted by distance
    """
    import math
    
    target_lab = rgb_to_lab(target_rgb)
    matches = []
    
    for code, rgb in PANTONE_TO_RGB.items():
        # Calculate LAB color difference
        lab = rgb_to_lab(rgb)
        
        # Calculate deltaE color difference
        L_diff = target_lab[0] - lab[0]
        a_diff = target_lab[1] - lab[1]
        b_diff = target_lab[2] - lab[2]
        
        distance = math.sqrt(L_diff**2 + a_diff**2 + b_diff**2)
        
        if distance <= tolerance:
            matches.append((code, rgb, distance))
    
    # Sort by distance (closest first)
    return sorted(matches, key=lambda x: x[2])

# For compatibility with existing code
TPX_COLORS = {}
TPG_COLORS = {}

def get_color_from_code(hex_code):
    """
    Converts a hex color code to BGR format for OpenCV.
    
    Args:
        hex_code (str): Hex color code like '#FF0000'
        
    Returns:
        tuple: Color in BGR format
    """
    # Remove the # if present
    if hex_code.startswith('#'):
        hex_code = hex_code[1:]
    
    # Convert hex to RGB
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    
    # Return as BGR for OpenCV
    return (b, g, r)
