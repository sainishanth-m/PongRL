"""
Reusable Metric Cards and UI components for PongRL Pro Dashboard.
"""
import streamlit as st
from typing import Optional


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    icon: str = "",
    subtitle: Optional[str] = None
):
    """
    Renders a modern glassmorphic metric card.
    """
    delta_html = ""
    if delta is not None:
        is_positive = False
        try:
            val_num = float(str(delta).replace("+", "").replace("%", "").strip())
            is_positive = val_num >= 0
        except ValueError:
            is_positive = "+" in str(delta)

        color_class = "positive" if (
            (delta_color == "normal" and is_positive) or 
            (delta_color == "inverse" and not is_positive)
        ) else "negative"
        
        delta_html = f"<div class='metric-delta {color_class}'>{delta}</div>"

    sub_html = f"<div class='metric-subtitle'>{subtitle}</div>" if subtitle else ""

    st.markdown(f"""
    <div class="glass-card metric-container">
        <div class="metric-header">
            <span class="metric-icon">{icon}</span>
            <span class="metric-label">{label}</span>
        </div>
        <div class="metric-value">{value}</div>
        {delta_html}
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_status_pill(status: str, variant: str = "success"):
    """Renders a status badge (success, warning, info, error)."""
    colors = {
        "success": ("rgba(0, 255, 170, 0.15)", "#00ffaa", "rgba(0, 255, 170, 0.3)"),
        "warning": ("rgba(255, 190, 11, 0.15)", "#ffbe0b", "rgba(255, 190, 11, 0.3)"),
        "danger": ("rgba(255, 0, 110, 0.15)", "#ff006e", "rgba(255, 0, 110, 0.3)"),
        "info": ("rgba(0, 212, 255, 0.15)", "#00d4ff", "rgba(0, 212, 255, 0.3)"),
    }
    bg, fg, border = colors.get(variant, colors["info"])
    return f"""
    <span style="
        background: {bg};
        color: {fg};
        border: 1px solid {border};
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        display: inline-block;
    ">{status}</span>
    """
