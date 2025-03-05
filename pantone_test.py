"""
Simple test script for the Pantone Color Extraction functionality.
"""

import streamlit as st
from pantone_tab import pantone_extraction_tab

# Set page configuration
st.set_page_config(
    page_title="Pantone Color Extraction Test",
    page_icon="🎨",
    layout="wide"
)

# Title
st.markdown("<h1 style='text-align: center;'>Pantone Color Extraction Test</h1>", unsafe_allow_html=True)

# Call the Pantone extraction tab function
pantone_extraction_tab()

# Footer
st.markdown("---")
st.markdown("This is a test of the Pantone Color Extraction functionality for ColorSep.")
