"""
Capital Flows Research Platform - Optimized Entry Point
Streamlit Community Cloud Deployment Version with Lazy Loading
"""

import sys
import os
from pathlib import Path

# Add source directories to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
dashboard_dir = src_dir / 'dashboard'

# Ensure all required directories are in path
for path in [str(src_dir), str(dashboard_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Set environment variable for streamlit optimization
os.environ['STREAMLIT_THEME_PRIMARY_COLOR'] = '#1f77b4'

if __name__ == "__main__":
    # Import and run the optimized main app
    from main_app import main
    
    try:
        main()
    except Exception as e:
        import streamlit as st
        st.error(f"Application Error: {e}")
        st.markdown("""
        ### Troubleshooting
        - Ensure all data files are present in `updated_data/Clean/`
        - Check that Python packages are installed: `pip install -r requirements.txt`
        - Refresh the page to retry loading
        """)