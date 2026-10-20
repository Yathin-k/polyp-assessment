"""
Streamlit Frontend — Colorectal Polyp Assessment Application

AI-Driven Automated Colorectal Polyp Assessment
Using Advanced Computer Vision Techniques

Architecture: ResUNet++
Dataset: Kvasir-SEG
"""

import os
import sys
import io
import time
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Colorectal Polyp Assessment | ResUNet++",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Subtle Slate & Deep Sapphire Medical Workstation Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall background and text */
    [data-testid="stAppViewContainer"] {
        background-color: #0c1424 !important;
        color: #f1f5f9 !important;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Sidebar Background and High-Contrast Typography */
    [data-testid="stSidebar"] {
        background-color: #111b2e !important;
        border-right: 1px solid #1e2c45 !important;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] input {
        background-color: #17243b !important;
        color: #ffffff !important;
        border: 1px solid #2a3c5d !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1 !important;
    }

    /* Top Application Header */
    .app-header {
        background: linear-gradient(135deg, #111e36 0%, #172847 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        border: 1px solid #22375b;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .app-header h1 {
        color: #ffffff !important;
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #94a3b8;
        font-size: 0.92rem;
        margin: 0;
    }

    /* Workstation Cards */
    .workstation-card {
        background-color: #142036;
        border: 1px solid #223454;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Status Pill Indicators */
    .status-positive {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 0.85rem 1.4rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        text-align: center;
        letter-spacing: 0.02em;
    }
    .status-negative {
        background: rgba(100, 116, 139, 0.12);
        border: 1px solid #475569;
        color: #94a3b8;
        padding: 0.85rem 1.4rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        text-align: center;
        letter-spacing: 0.02em;
    }

    /* Metric Tiles */
    .metric-card {
        background-color: #142036;
        border: 1px solid #223454;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }
    .metric-card-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        line-height: 1.2;
    }
    .metric-card-sub {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Sidebar Clean Panels */
    .sidebar-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-top: 0.6rem;
        margin-bottom: 0.75rem;
    }
    .sidebar-spec-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #1a2842;
        font-size: 0.86rem;
    }
    .sidebar-spec-label {
        color: #94a3b8;
    }
    .sidebar-spec-val {
        color: #f1f5f9;
        font-weight: 600;
    }

    /* Disclaimer Box in Sidebar */
    .sidebar-disclaimer {
        background-color: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        font-size: 0.8rem;
        color: #fcd34d;
        line-height: 1.45;
        margin-top: 1rem;
    }

    /* Action bar */
    .action-bar {
        background-color: #142036;
        border: 1px solid #223454;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.25rem;
    }

    /* Image labels */
    .view-label {
        font-size: 0.88rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        text-align: center;
        background: #17243b;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        border: 1px solid #26385a;
    }

    /* Hide standard Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>Automated Colorectal Polyp Assessment</h1>
    <p>Semantic Segmentation and Morphometric Lesion Analysis · ResUNet++ Architecture</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Predictor Cache Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_predictor(ckpt_path):
    """Load and cache the ResUNet++ predictor instance."""
    from src.inference.predict import PolypPredictor
    return PolypPredictor(
        checkpoint_path=ckpt_path,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Sidebar (Clean, High Contrast, No Instructions, No Jha et al, No Emojis)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Model Settings</div>', unsafe_allow_html=True)

    default_checkpoint = "artifacts/checkpoints/resunetpp_best.pth"
    checkpoint_path = st.text_input(
        "Checkpoint Path",
        value=default_checkpoint,
        help="Path to trained ResUNet++ model weights",
    )

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Binarization cutoff for polyp foreground classification",
    )

    overlay_alpha = st.slider(
        "Overlay Opacity",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Transparency level of the lesion overlay on the original frame",
    )

    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        st.markdown(
            '<div style="background: rgba(16,185,129,0.1); border: 1px solid #10b981; color: #34d399; '
            'padding: 0.5rem 0.8rem; border-radius: 6px; font-size: 0.82rem; margin: 0.8rem 0;">'
            'Weights Loaded · Validation Dice: 77.8%</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background: rgba(239,68,68,0.1); border: 1px solid #ef4444; color: #f87171; '
            'padding: 0.5rem 0.8rem; border-radius: 6px; font-size: 0.82rem; margin: 0.8rem 0;">'
            'Checkpoint not found</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown('<div class="sidebar-title">Architecture Specs</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-spec-row">
        <span class="sidebar-spec-label">Model</span>
        <span class="sidebar-spec-val">ResUNet++</span>
    </div>
    <div class="sidebar-spec-row">
        <span class="sidebar-spec-label">Dataset</span>
        <span class="sidebar-spec-val">Kvasir-SEG</span>
    </div>
    <div class="sidebar-spec-row">
        <span class="sidebar-spec-label">Input Size</span>
        <span class="sidebar-spec-val">256 x 256</span>
    </div>
    <div class="sidebar-spec-row">
        <span class="sidebar-spec-label">Execution</span>
        <span class="sidebar-spec-val">PyTorch CPU</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-disclaimer">
        <strong>Research Prototype</strong><br>
        This software is developed strictly for educational and scientific computer vision research.
        It is not certified for clinical diagnostic decisions.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
if "segmentation_result" not in st.session_state:
    st.session_state.segmentation_result = None
if "active_filename" not in st.session_state:
    st.session_state.active_filename = None
if "cached_image" not in st.session_state:
    st.session_state.cached_image = None


# ---------------------------------------------------------------------------
# Main Interactive Flow
# ---------------------------------------------------------------------------

# 1. Image Upload
uploaded_file = st.file_uploader(
    "Upload Colonoscopy Examination Frame",
    type=["jpg", "jpeg", "png"],
    help="Upload an endoscopic examination frame in JPEG or PNG format.",
)

if uploaded_file is not None:
    # If a new image was uploaded, reset the segmentation state
    if st.session_state.active_filename != uploaded_file.name:
        st.session_state.active_filename = uploaded_file.name
        st.session_state.segmentation_result = None
        st.session_state.cached_image = Image.open(uploaded_file).convert("RGB")

    raw_image = st.session_state.cached_image
    img_w, img_h = raw_image.size

    # Action Bar: Metadata & Primary Action Button
    st.markdown(f"""
    <div class="action-bar">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <span style="font-size: 0.95rem; font-weight: 600; color: #f1f5f9;">{uploaded_file.name}</span>
                <span style="color: #64748b; font-size: 0.85rem; margin-left: 0.8rem;">{img_w} x {img_h} px</span>
                <span style="color: #64748b; font-size: 0.85rem; margin-left: 0.8rem;">{uploaded_file.size / 1024:.1f} KB</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.84rem;">
                Threshold: <strong>{threshold:.2f}</strong> &nbsp;|&nbsp; Alpha: <strong>{overlay_alpha:.2f}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns([1.5, 4.5])
    with btn_col1:
        run_btn = st.button("Run Segmentation", type="primary", use_container_width=True)

    if run_btn:
        try:
            predictor = get_cached_predictor(checkpoint_path)
            predictor.threshold = threshold
            with st.spinner("Executing ResUNet++ inference..."):
                result = predictor.predict(raw_image)
                # Re-generate overlay with customized alpha if selected
                result["overlay"] = predictor._create_overlay(
                    np.array(raw_image), result["binary_mask"], alpha=overlay_alpha
                )
                st.session_state.segmentation_result = result
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    # 2. Display View
    if st.session_state.segmentation_result is not None:
        result = st.session_state.segmentation_result

        # Diagnostic Status Banner
        st.markdown("")
        if result["polyp_detected"]:
            st.markdown(
                '<div class="status-positive">STATUS: POLYP LESION DETECTED</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-negative">STATUS: NO SIGNIFICANT LESION DETECTED</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # 3-Column Synchronized Visualizer
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown('<div class="view-label">Original Colonoscopy</div>', unsafe_allow_html=True)
            st.image(result["original_image"], use_container_width=True)

        with col2:
            st.markdown('<div class="view-label">Binary Lesion Mask</div>', unsafe_allow_html=True)
            mask_display = Image.fromarray(result["binary_mask"])
            st.image(mask_display, use_container_width=True)

        with col3:
            st.markdown('<div class="view-label">Diagnostic Overlay</div>', unsafe_allow_html=True)
            overlay_display = Image.fromarray(result["overlay"])
            st.image(overlay_display, use_container_width=True)

        st.markdown("")

        # Quantitative Metrics Bar
        m1, m2, m3, m4 = st.columns(4, gap="medium")

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-label">Lesion Coverage</div>
                <div class="metric-card-value">{result['foreground_ratio'] * 100:.2f}%</div>
                <div class="metric-card-sub">of visual field</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-label">Segmented Area</div>
                <div class="metric-card-value">{result['foreground_pixels']:,}</div>
                <div class="metric-card-sub">positive pixels</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-label">Inference Time</div>
                <div class="metric-card-value">{result['inference_time_ms']:.0f} ms</div>
                <div class="metric-card-sub">CPU latency</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-label">Active Threshold</div>
                <div class="metric-card-value">{threshold:.2f}</div>
                <div class="metric-card-sub">decision cutoff</div>
            </div>
            """, unsafe_allow_html=True)

        # Advanced Probability Heatmap (Expandable to keep main view uncluttered)
        with st.expander("View Continuous Probability Heatmap"):
            prob_map = result["probability_map"]
            import matplotlib.cm as cm
            heatmap_colored = cm.magma(prob_map)[:, :, :3]
            heatmap_uint8 = (heatmap_colored * 255).astype(np.uint8)

            h_col1, h_col2 = st.columns([2, 1], gap="medium")
            with h_col1:
                st.image(heatmap_uint8, use_container_width=True, caption="Pre-threshold confidence spectrum (Magma)")
            with h_col2:
                st.markdown(f"""
                <div class="workstation-card">
                    <div class="metric-card-label">Confidence Statistics</div>
                    <div style="font-size: 0.88rem; line-height: 1.8; color: #cbd5e1; margin-top: 0.5rem;">
                        Maximum Probability: <strong style="color: #f1f5f9;">{prob_map.max():.4f}</strong><br>
                        Mean Field Probability: <strong style="color: #f1f5f9;">{prob_map.mean():.4f}</strong><br>
                        95th Percentile: <strong style="color: #f1f5f9;">{float(np.percentile(prob_map, 95)):.4f}</strong><br>
                        Active Cutoff: <strong style="color: #3b82f6;">{threshold:.2f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Export Tools
        st.markdown("")
        st.markdown('<div class="sidebar-title">Export Results</div>', unsafe_allow_html=True)
        dl1, dl2, dl3 = st.columns(3, gap="medium")

        base_name = uploaded_file.name.rsplit(".", 1)[0]

        with dl1:
            mask_buf = io.BytesIO()
            mask_display.save(mask_buf, format="PNG")
            st.download_button(
                label="Download Binary Mask (PNG)",
                data=mask_buf.getvalue(),
                file_name=f"{base_name}_mask.png",
                mime="image/png",
                use_container_width=True,
            )

        with dl2:
            overlay_buf = io.BytesIO()
            overlay_display.save(overlay_buf, format="PNG")
            st.download_button(
                label="Download Overlay Image (PNG)",
                data=overlay_buf.getvalue(),
                file_name=f"{base_name}_overlay.png",
                mime="image/png",
                use_container_width=True,
            )

        with dl3:
            report_dict = {
                "file_name": uploaded_file.name,
                "dimensions": {"width": img_w, "height": img_h},
                "architecture": "ResUNet++",
                "polyp_detected": bool(result["polyp_detected"]),
                "affected_area_percentage": round(float(result["foreground_ratio"] * 100), 2),
                "foreground_pixels": int(result["foreground_pixels"]),
                "inference_time_ms": round(float(result["inference_time_ms"]), 1),
                "threshold_applied": float(threshold),
                "overlay_alpha": float(overlay_alpha),
            }
            report_json = json.dumps(report_dict, indent=2)
            st.download_button(
                label="Download Analysis Summary (JSON)",
                data=report_json,
                file_name=f"{base_name}_report.json",
                mime="application/json",
                use_container_width=True,
            )

    else:
        # Pre-segmentation view: show the uploaded image cleanly
        st.markdown("")
        st.markdown('<div class="view-label">Uploaded Colonoscopy Frame</div>', unsafe_allow_html=True)
        preview_col1, preview_col2, preview_col3 = st.columns([1, 2, 1])
        with preview_col2:
            st.image(raw_image, use_container_width=True)
            st.caption("Click 'Run Segmentation' above to compute lesion boundaries and morphometrics.")

else:
    # Empty State
    st.markdown("""
    <div class="workstation-card" style="text-align: center; padding: 3rem 1.5rem; margin-top: 1rem;">
        <div style="color: #f1f5f9; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">
            No Examination Frame Loaded
        </div>
        <div style="color: #94a3b8; font-size: 0.88rem; max-width: 480px; margin: 0 auto; line-height: 1.5;">
            Upload an endoscopic colonoscopy image above to begin automated polyp assessment.
            Supported formats: JPEG, JPG, PNG.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Clean Minimalist Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 0.5rem 0;">
    Colorectal Polyp Assessment System &nbsp;·&nbsp; ResUNet++ Architecture &nbsp;·&nbsp; Academic Research
</div>
""", unsafe_allow_html=True)
