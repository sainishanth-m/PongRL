"""
Custom CSS styles and themes for PongRL Pro Dashboard.
"""
import streamlit as st

DARK_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1024 0%, #070812 90%);
        color: #e4e7f5;
    }
    
    /* Headers and Gradients */
    .hero-title {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #9aa0bc;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 24px;
    }
    
    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(18, 22, 44, 0.72);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(0, 212, 255, 0.28);
        box-shadow: 0 14px 36px rgba(0, 212, 255, 0.08);
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 120px;
    }
    .metric-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .metric-icon {
        font-size: 1.25rem;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #9aa0bc;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 6px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .metric-delta.positive {
        color: #00ffaa;
    }
    .metric-delta.negative {
        color: #ff006e;
    }
    .metric-subtitle {
        color: #727792;
        font-size: 0.76rem;
        margin-top: 4px;
    }
    
    /* Sidebar Styling */
    .sidebar-divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.08);
        margin: 16px 0;
    }
    .sidebar-info-box {
        background: rgba(13, 16, 32, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px 14px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .sidebar-info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.82rem;
    }
    .info-dim {
        color: #787d9a;
    }
    .info-val {
        color: #e2e4ef;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
    
    /* Code and Preformatted */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(10, 13, 28, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
    }
    
    /* Scoreboard Box */
    .scoreboard-box {
        background: linear-gradient(135deg, rgba(20, 25, 52, 0.9), rgba(12, 15, 33, 0.9));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .score-title {
        font-size: 0.85rem;
        color: #8e92a8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .score-digits {
        font-size: 2.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #00d4ff;
        margin: 4px 0;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
"""

def inject_custom_styles():
    """Injects dark theme CSS into the current Streamlit app."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

