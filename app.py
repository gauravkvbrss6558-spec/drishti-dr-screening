"""
Explainable AI for Diabetic Retinopathy Screening — Demo Interface
SIH26038 · MathWorks · Clean & Green Technology

Run locally after placing `dr_model_final.pth` and `model_metadata.json`
(exported from the Kaggle notebook) in the same folder as this file, and
keeping the `assets/` folder (with eye_anatomy.png) alongside it too:

    pip install -r requirements.txt
    streamlit run app.py

Designed for non-specialist healthcare workers (e.g. ASHA workers):
one button, one image, one clear result, one explanation heatmap.

Multi-language: UI strings live in translations.py and are looked up via
t("key"). See that file's docstring for how to add more languages.
"""

import csv
import io
import json
import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import torch.nn.functional as F
from fpdf import FPDF
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from translations import t, LANGUAGES

# ----------------------------------------------------------------------------
# Fix: st.markdown(unsafe_allow_html=True) renders raw HTML as literal text
# whenever a line starts with 4+ spaces (Markdown treats that as a fenced
# code block). Our multi-line f-strings are indented to match the
# surrounding Python code, so every such call is affected.
#
# textwrap.dedent() alone isn't enough here: several blocks interpolate
# EYE_MOTIF_SVG, which is itself an unindented (column-0) string, so mixing
# it in drops the *common* leading whitespace to zero and dedent ends up
# stripping nothing from the wrapper <div> lines. Instead, strip leading
# whitespace from every line individually — safe for plain HTML since it
# doesn't use <pre>/<code> blocks that rely on that whitespace.
# ----------------------------------------------------------------------------
_original_markdown = st.markdown


def _dedented_markdown(body, *args, **kwargs):
    if isinstance(body, str):
        body = "\n".join(line.lstrip() for line in body.split("\n"))
    return _original_markdown(body, *args, **kwargs)


st.markdown = _dedented_markdown

# ----------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Drishti — DR Screening Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Language must be initialized before anything else reads t(), since t()
# reads st.session_state.lang.
if "lang" not in st.session_state:
    st.session_state.lang = "en"

MODEL_PATH = "dr_model_final.pth"
METADATA_PATH = "model_metadata.json"
EYE_DIAGRAM_PATH = "assets/eye_anatomy.png"

# Local CSV log of every screening, keyed by patient ID/name — lets a health
# worker look up a patient's past screenings without re-typing anything.
RECORDS_PATH = "patient_records.csv"
RECORDS_FIELDS = [
    "timestamp", "patient_id", "patient_name",
    "grade", "severity_label", "confidence",
]

# Recommendations and severity labels are looked up via t("rec_<idx>") /
# t("grade_<idx>") / t("risk_<key>") so they follow the selected language.
SEVERITY_STYLE = {
    0: {"color": "#1B7A43", "bg": "#EAF6EE", "label_key": "risk_low",      "icon": "✅"},
    1: {"color": "#8A6D1B", "bg": "#F6F1E1", "label_key": "risk_mild",     "icon": "🟡"},
    2: {"color": "#B8720B", "bg": "#FBEEDA", "label_key": "risk_moderate", "icon": "🟠"},
    3: {"color": "#C2540A", "bg": "#FCE7D9", "label_key": "risk_high",     "icon": "🔶"},
    4: {"color": "#B3261E", "bg": "#FBE1DF", "label_key": "risk_critical", "icon": "🔴"},
}

# --- Design tokens ------------------------------------------------------
# Palette pulled from the anatomy of the eye itself: deep navy sclera-shadow,
# a warm coral/amber iris tone, and a teal optic-nerve accent.
INK = "#0B1E33"          # deep navy — headlines, hero background
INK_SOFT = "#334155"     # secondary text
CANVAS = "#FAF6F0"       # warm paper background (not stark white)
CARD = "#FFFFFF"
CORAL = "#FF7A50"        # primary accent — iris warmth
CORAL_DARK = "#D9542E"
TEAL = "#2FBFA6"         # secondary accent — optic nerve
TEAL_SOFT = "#E6F7F3"
BORDER = "#E9E1D4"
MUTED = "#6B7A8C"

# Below this confidence, the model's top prediction is not decisively separated
# from the alternatives — flag it so a human reviewer knows to double-check.
LOW_CONFIDENCE_THRESHOLD = 65.0

# Below this variance-of-Laplacian score, the image is considered too blurry
# to screen reliably. Lower values = fewer sharp edges = blurrier image.
# 100 is a common general-purpose default; tune it against real fundus photos
# from your camera/devices if you see false positives or negatives.
BLUR_THRESHOLD = 100.0


# ----------------------------------------------------------------------------
# A small hand-drawn eye motif, used as the hero's centerpiece visual instead
# of a stock photo or emoji — an iris rendered in the app's own palette.
# ----------------------------------------------------------------------------
EYE_MOTIF_SVG = """
<svg width="240" height="240" viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg" class="eye-motif-svg">
  <defs>
    <radialGradient id="irisGrad" cx="32%" cy="28%" r="78%">
      <stop offset="0%" stop-color="#FFE3C9"/>
      <stop offset="28%" stop-color="#FFA26B"/>
      <stop offset="60%" stop-color="#FF7A50"/>
      <stop offset="100%" stop-color="#8A2410"/>
    </radialGradient>
    <radialGradient id="glowGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#2FBFA6" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#2FBFA6" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="scleraGrad" cx="40%" cy="30%" r="80%">
      <stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="55%" stop-color="#F7F1E6"/>
      <stop offset="100%" stop-color="#E4D9C5"/>
    </radialGradient>
    <radialGradient id="pupilGrad" cx="38%" cy="32%" r="70%">
      <stop offset="0%" stop-color="#1C3A57"/>
      <stop offset="100%" stop-color="#03090F"/>
    </radialGradient>
    <linearGradient id="lidSheen" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0"/>
    </linearGradient>
    <filter id="softDrop" x="-40%" y="-40%" width="180%" height="180%">
      <feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#03121F" flood-opacity="0.35"/>
    </filter>
  </defs>
  <circle cx="110" cy="110" r="108" fill="url(#glowGrad)"/>
  <g filter="url(#softDrop)">
    <path d="M8 110 Q110 28 212 110 Q110 192 8 110 Z" fill="url(#scleraGrad)"/>
    <circle cx="110" cy="110" r="57" fill="url(#irisGrad)"/>
    <g stroke="#7A1F0C" stroke-width="1" opacity="0.32">
      <line x1="110" y1="62" x2="110" y2="88"/>
      <line x1="110" y1="132" x2="110" y2="158"/>
      <line x1="62" y1="110" x2="88" y2="110"/>
      <line x1="132" y1="110" x2="158" y2="110"/>
      <line x1="76" y1="76" x2="94" y2="94"/>
      <line x1="144" y1="76" x2="126" y2="94"/>
      <line x1="76" y1="144" x2="94" y2="126"/>
      <line x1="144" y1="144" x2="126" y2="126"/>
    </g>
    <circle cx="110" cy="110" r="25" fill="url(#pupilGrad)"/>
    <circle cx="98" cy="96" r="8" fill="#FFFFFF" opacity="0.95"/>
    <circle cx="122" cy="122" r="3" fill="#FFFFFF" opacity="0.4"/>
  </g>
  <path d="M8 110 Q110 22 212 110 Q110 60 8 110 Z" fill="url(#lidSheen)"/>
  <path d="M8 110 Q110 28 212 110" fill="none" stroke="#0B1E33" stroke-width="3.5" opacity="0.6" stroke-linecap="round"/>
  <path d="M8 110 Q110 192 212 110" fill="none" stroke="#0B1E33" stroke-width="3.5" opacity="0.4" stroke-linecap="round"/>
</svg>
"""


# ----------------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');

            /* --------------------------------------------------------------
               Force a light color scheme everywhere. Without this, a phone
               or browser set to system dark mode makes Streamlit switch its
               *native* widgets (selectbox, expander, etc.) to a dark theme,
               while our custom HTML/CSS still assumes a fixed light/dark
               split — producing invisible near-white-on-white or
               near-black-on-black text in a few spots (patient records
               list, language selector). Forcing color-scheme stops browsers
               from auto-darkening native form controls.
               -------------------------------------------------------------- */
            :root, html, body {{
                color-scheme: light only;
            }}

            html, body, [class*="css"] {{
                font-family: 'Inter', 'Noto Sans Devanagari', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            .stApp {{
                background:
                    radial-gradient(ellipse 700px 500px at 8% -8%, rgba(255,122,80,0.14), transparent 60%),
                    radial-gradient(ellipse 600px 500px at 96% 12%, rgba(47,191,166,0.14), transparent 55%),
                    radial-gradient(ellipse 800px 600px at 50% 110%, rgba(11,30,51,0.06), transparent 60%),
                    {CANVAS};
                perspective: 1800px;
            }}

            .block-container {{
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1080px;
            }}

            h1, h2, h3, .headline {{
                font-family: 'Fraunces', 'Noto Sans Devanagari', Georgia, serif;
            }}

            /* Floating animation shared by hero art */
            @keyframes floatY {{
                0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                50% {{ transform: translateY(-9px) rotate(1.2deg); }}
            }}
            @keyframes shine-sweep {{
                0% {{ transform: translateX(-120%) rotate(8deg); }}
                100% {{ transform: translateX(220%) rotate(8deg); }}
            }}
            /* True 3D sphere-like rotation for the hero eye — real perspective + rotateX/Y,
               not just a 2D float, so the eyeball genuinely reads as an object in space. */
            @keyframes eye3d {{
                0%   {{ transform: perspective(900px) rotateY(-14deg) rotateX(5deg)  translateY(0px)   translateZ(0px); }}
                50%  {{ transform: perspective(900px) rotateY(14deg)  rotateX(-5deg) translateY(-12px) translateZ(30px); }}
                100% {{ transform: perspective(900px) rotateY(-14deg) rotateX(5deg)  translateY(0px)   translateZ(0px); }}
            }}
            /* Slow ambient drift for the glass orbs floating in the hero background */
            @keyframes floatSlow {{
                0%, 100% {{ transform: translateY(0px) translateX(0px) scale(1); }}
                50% {{ transform: translateY(-18px) translateX(8px) scale(1.06); }}
            }}

            /* Hero — layered like a raised panel: dark base, inner sheen, outer contact shadow */
            .hero {{
                background:
                    radial-gradient(circle at 15% 15%, rgba(255,122,80,0.16), transparent 45%),
                    radial-gradient(circle at 85% 85%, rgba(47,191,166,0.14), transparent 50%),
                    linear-gradient(155deg, #17395C 0%, {INK} 48%, #071120 100%);
                border-radius: 26px;
                padding: 2.8rem 3rem;
                margin-bottom: 2rem;
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow:
                    0 2px 0 rgba(255,255,255,0.06) inset,
                    0 -30px 60px rgba(0,0,0,0.25) inset,
                    0 18px 30px rgba(7, 17, 32, 0.35),
                    0 40px 70px rgba(7, 17, 32, 0.3);
                position: relative;
                overflow: hidden;
                transform-style: preserve-3d;
            }}
            .hero::before {{
                /* glossy top highlight strip, like light hitting curved glass */
                content: "";
                position: absolute;
                top: 0; left: 0; right: 0;
                height: 50%;
                background: linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0) 100%);
                pointer-events: none;
            }}
            .hero-inner {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 2.2rem;
                position: relative;
                z-index: 1;
            }}
            .hero-copy {{
                flex: 1;
                min-width: 260px;
            }}
            .hero-tag {{
                display: inline-block;
                color: {CORAL};
                font-size: 0.82rem;
                font-weight: 600;
                margin-bottom: 0.9rem;
                letter-spacing: 0.01em;
                text-shadow: 0 1px 8px rgba(255,122,80,0.4);
            }}
            .hero h1 {{
                color: #FBF7F0;
                font-size: 2.6rem;
                font-weight: 600;
                margin: 0 0 0.7rem 0;
                line-height: 1.12;
                letter-spacing: -0.01em;
                text-shadow: 0 2px 18px rgba(0,0,0,0.35);
            }}
            .hero h1 em {{
                color: {CORAL};
                font-style: normal;
            }}
            .hero p {{
                color: #C7D2DE;
                font-size: 1.02rem;
                margin: 0;
                max-width: 480px;
                line-height: 1.6;
            }}
            .hero-eye {{
                flex-shrink: 0;
                animation: eye3d 7s ease-in-out infinite;
                transform-style: preserve-3d;
                filter: drop-shadow(0 20px 25px rgba(0,0,0,0.4));
            }}
            @media (max-width: 700px) {{
                .hero-inner {{ flex-direction: column; text-align: center; }}
                .hero p {{ max-width: 100%; }}
            }}

            /* Ambient glass orbs — blurred, translucent spheres drifting behind the hero
               copy to add a layer of depth in front of / behind the gradient background. */
            .glass-orb {{
                position: absolute;
                border-radius: 50%;
                pointer-events: none;
                filter: blur(1px);
                animation: floatSlow 9s ease-in-out infinite;
                box-shadow: 0 20px 45px rgba(0,0,0,0.3), inset 0 0 20px rgba(255,255,255,0.15);
            }}
            .glass-orb.orb1 {{
                width: 110px; height: 110px; top: 8%; right: 14%;
                background: radial-gradient(circle at 32% 28%, rgba(47,191,166,0.55), rgba(47,191,166,0.05) 70%);
                animation-delay: 0s;
            }}
            .glass-orb.orb2 {{
                width: 60px; height: 60px; bottom: 14%; left: 8%;
                background: radial-gradient(circle at 32% 28%, rgba(255,122,80,0.5), rgba(255,122,80,0.05) 70%);
                animation-delay: 2.4s;
            }}
            .glass-orb.orb3 {{
                width: 34px; height: 34px; top: 55%; right: 30%;
                background: radial-gradient(circle at 32% 28%, rgba(255,255,255,0.45), rgba(255,255,255,0.02) 70%);
                animation-delay: 4.6s;
            }}

            /* Elements tagged data-tilt get a live perspective tilt driven by the mouse
               (see inject_tilt_script) — this just prepares them to receive it smoothly. */
            [data-tilt] {{
                transform-style: preserve-3d;
                will-change: transform;
                transition: transform 0.4s cubic-bezier(0.22, 1, 0.36, 1);
            }}

            /* Upload zone card — raised "tray" with dashed inset well */
            .upload-card {{
                background: linear-gradient(180deg, #FFFFFF 0%, #FFFCF8 100%);
                border-radius: 20px;
                padding: 0.55rem;
                margin-bottom: 1.6rem;
                border: 1px solid {BORDER};
                box-shadow:
                    0 1px 0 rgba(255,255,255,0.9) inset,
                    0 14px 24px rgba(11, 30, 51, 0.08),
                    0 4px 8px rgba(11, 30, 51, 0.05);
                transition: transform 0.25s ease, box-shadow 0.25s ease;
            }}
            .upload-card:hover {{
                transform: translateY(-4px);
                box-shadow:
                    0 1px 0 rgba(255,255,255,0.9) inset,
                    0 22px 34px rgba(11, 30, 51, 0.12),
                    0 6px 12px rgba(11, 30, 51, 0.06);
            }}
            .upload-card [data-testid="stFileUploaderDropzone"] {{
                border: 1.5px dashed {CORAL} !important;
                border-radius: 14px !important;
                background: repeating-linear-gradient(135deg, rgba(255,122,80,0.035) 0px, rgba(255,122,80,0.035) 10px, transparent 10px, transparent 20px), #FFFEFB !important;
            }}
            .upload-card [data-testid="stBaseButton-secondary"] {{
                background: linear-gradient(180deg, #FFFFFF 0%, #F3ECE1 100%) !important;
                border: 1px solid {BORDER} !important;
                border-radius: 10px !important;
                box-shadow: 0 3px 0 {BORDER}, 0 4px 8px rgba(11,30,51,0.1) !important;
                transition: transform 0.15s ease, box-shadow 0.15s ease !important;
            }}
            .upload-card [data-testid="stBaseButton-secondary"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 0 {BORDER}, 0 8px 14px rgba(11,30,51,0.14) !important;
            }}
            .upload-card [data-testid="stBaseButton-secondary"]:active {{
                transform: translateY(1px);
                box-shadow: 0 1px 0 {BORDER}, 0 2px 4px rgba(11,30,51,0.1) !important;
            }}

            /* File vs. camera tabs inside the upload card */
            .upload-card [data-testid="stTabs"] [data-baseweb="tab-list"] {{
                gap: 0.4rem;
                border-bottom: 1px solid {BORDER};
            }}
            .upload-card [data-testid="stTabs"] button[data-baseweb="tab"] {{
                font-weight: 600;
                font-size: 0.88rem;
                color: {MUTED};
            }}
            .upload-card [data-testid="stTabs"] button[aria-selected="true"] {{
                color: {CORAL_DARK};
            }}
            .upload-card [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
                background-color: {CORAL} !important;
            }}
            /* Camera capture widget — round action button + rounded video frame,
               matching the dashed-well treatment used for the file dropzone. */
            .upload-card [data-testid="stCameraInput"] video,
            .upload-card [data-testid="stCameraInput"] img {{
                border-radius: 14px !important;
                border: 1.5px solid {BORDER} !important;
            }}
            .upload-card [data-testid="stCameraInput"] [data-testid="stCameraInputWindow"] {{
                border-radius: 14px !important;
                overflow: hidden;
                background: #FFFEFB !important;
            }}
            .upload-card [data-testid="stCameraInputButton"] button,
            .upload-card [data-testid="stCameraInput"] button {{
                background: linear-gradient(180deg, #FF8F6B 0%, {CORAL} 55%, {CORAL_DARK} 100%) !important;
                border: none !important;
                color: white !important;
                font-weight: 600 !important;
                border-radius: 12px !important;
                box-shadow: 0 1px 0 rgba(255,255,255,0.35) inset, 0 4px 0 {CORAL_DARK}, 0 10px 18px rgba(217,84,46,0.35) !important;
            }}
            .upload-hint {{
                display: flex;
                align-items: center;
                gap: 0.6rem;
                color: {MUTED};
                font-size: 0.86rem;
                padding: 0.3rem 0.6rem 0.7rem 0.6rem;
            }}

            /* Patient ID / Patient Name input labels — force a visible,
               theme-independent color. Without this, on a phone set to dark
               mode Streamlit auto-switches to its dark theme and these
               labels render near-white on our light upload-card background,
               making them unreadable. */
            [data-testid="stTextInput"] label,
            [data-testid="stTextInput"] label p,
            [data-testid="stWidgetLabel"] p {{
                color: {INK_SOFT} !important;
                font-weight: 600 !important;
                opacity: 1 !important;
            }}

            [data-testid="stTextInput"] input {{
                color: {INK} !important;
                background-color: #FFFFFF !important;
                border: 1px solid {BORDER} !important;
                border-radius: 10px !important;
            }}

            [data-testid="stTextInput"] input::placeholder {{
                color: {MUTED} !important;
                opacity: 1 !important;
            }}

            /* Language selector select box — keep it legible against the dark sidebar */
            [data-testid="stSidebar"] [data-testid="stSelectbox"] label p {{
                color: #E7EDF3 !important;
                font-weight: 700 !important;
            }}

            /* --------------------------------------------------------------
               The selectbox LABEL ("भाषा / Language") was already fixed
               above, but the *selected value* ("English") and dropdown
               options render through BaseWeb's own internals, which follow
               the browser/OS theme independently and were going
               near-invisible in dark mode. Force them explicitly, with a
               wide net of selectors since BaseWeb's own class names are
               randomly hashed per build and can't be targeted directly. */
            [data-testid="stSidebar"] [data-testid="stSelectbox"] > div,
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
                background-color: #FFFFFF !important;
                border-radius: 10px !important;
            }}
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] div,
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] span,
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"],
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"] * {{
                color: {INK} !important;
                fill: {INK} !important;
                -webkit-text-fill-color: {INK} !important;
                opacity: 1 !important;
            }}
            /* The open dropdown list renders in a portal appended to <body>,
               outside the sidebar entirely, so it needs its own top-level rule. */
            div[data-baseweb="popover"],
            div[data-baseweb="menu"],
            ul[role="listbox"] {{
                background-color: #FFFFFF !important;
            }}
            div[data-baseweb="popover"] *,
            div[data-baseweb="menu"] *,
            ul[role="listbox"] li,
            ul[role="listbox"] li * {{
                color: {INK} !important;
                -webkit-text-fill-color: {INK} !important;
                opacity: 1 !important;
            }}
            ul[role="listbox"] li:hover,
            ul[role="listbox"] li[aria-selected="true"] {{
                background-color: {TEAL_SOFT} !important;
            }}

            /* Section headers */
            .section-label {{
                font-size: 0.78rem;
                font-weight: 700;
                color: {INK_SOFT};
                margin-bottom: 0.6rem;
                border-left: 3px solid {TEAL};
                padding-left: 0.55rem;
            }}

            /* Result card — raised panel with a colored glow that matches severity */
            .result-card {{
                position: relative;
                border-radius: 20px;
                padding: 1.6rem 1.8rem;
                margin: 1.1rem 0;
                border-left: 6px solid var(--sev-color);
                background: linear-gradient(155deg, var(--sev-bg) 0%, {CARD} 130%);
                box-shadow:
                    0 1px 0 rgba(255,255,255,0.7) inset,
                    0 16px 30px -8px color-mix(in srgb, var(--sev-color) 35%, transparent),
                    0 6px 14px rgba(11,30,51,0.06);
                overflow: hidden;
                transition: transform 0.25s ease, box-shadow 0.25s ease;
            }}
            .result-card:hover {{
                transform: translateY(-3px) scale(1.003);
                box-shadow:
                    0 1px 0 rgba(255,255,255,0.7) inset,
                    0 22px 40px -8px color-mix(in srgb, var(--sev-color) 42%, transparent),
                    0 8px 18px rgba(11,30,51,0.08);
            }}
            .result-card::after {{
                content: "";
                position: absolute;
                top: -60%; left: -20%;
                width: 40%; height: 220%;
                background: linear-gradient(120deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.35) 50%, rgba(255,255,255,0) 100%);
                animation: shine-sweep 3.2s ease-in-out infinite;
                animation-delay: 0.4s;
                pointer-events: none;
            }}
            .result-title {{
                font-family: 'Fraunces', 'Noto Sans Devanagari', Georgia, serif;
                font-size: 1.65rem;
                font-weight: 600;
                color: var(--sev-color);
                margin: 0 0 0.15rem 0;
                text-shadow: 0 1px 0 rgba(255,255,255,0.5);
            }}
            .result-sub {{
                color: {INK};
                font-size: 0.92rem;
                opacity: 0.75;
                margin: 0;
            }}
            .confidence-pill {{
                display: inline-block;
                background: linear-gradient(180deg, #FFFFFF 0%, #F5F1EA 100%);
                border-radius: 20px;
                padding: 0.32rem 0.95rem;
                font-weight: 700;
                font-size: 0.85rem;
                color: var(--sev-color);
                margin-top: 0.75rem;
                box-shadow:
                    0 1px 0 rgba(255,255,255,0.8) inset,
                    0 3px 8px rgba(0,0,0,0.1),
                    0 1px 2px rgba(0,0,0,0.06);
                position: relative;
                z-index: 1;
            }}

            /* Recommendation box */
            .rec-box {{
                background: linear-gradient(155deg, {TEAL_SOFT} 0%, #FFFFFF 140%);
                border-radius: 16px;
                padding: 1.2rem 1.45rem;
                border-left: 4px solid {TEAL};
                color: {INK};
                font-size: 0.96rem;
                line-height: 1.55;
                box-shadow: 0 1px 0 rgba(255,255,255,0.8) inset, 0 10px 22px rgba(47,191,166,0.14);
            }}

            /* Image frames — give the two photos real card presence */
            div[data-testid="stImage"] {{
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 14px 28px rgba(11,30,51,0.16), 0 3px 8px rgba(11,30,51,0.08);
                border: 4px solid {CARD};
                transition: transform 0.25s ease, box-shadow 0.25s ease;
            }}
            div[data-testid="stImage"]:hover {{
                transform: translateY(-4px) scale(1.01);
                box-shadow: 0 22px 36px rgba(11,30,51,0.2), 0 4px 10px rgba(11,30,51,0.1);
            }}

            /* Image captions */
            .img-caption {{
                text-align: center;
                font-size: 0.85rem;
                font-weight: 600;
                color: {MUTED};
                margin-top: 0.7rem;
            }}

            /* Low-confidence warning banner */
            .low-conf-warning {{
                background: linear-gradient(155deg, #FDEDEC 0%, #FFF8F7 100%);
                border: 1.5px solid #E8A6A0;
                border-radius: 16px;
                padding: 1.05rem 1.35rem;
                margin: 1rem 0;
                display: flex;
                gap: 0.7rem;
                align-items: flex-start;
                box-shadow: 0 1px 0 rgba(255,255,255,0.7) inset, 0 12px 22px rgba(194,54,10,0.1);
            }}
            .low-conf-warning .lcw-title {{
                font-weight: 700;
                color: #9B2C24;
                font-size: 0.92rem;
                margin-bottom: 0.2rem;
            }}
            .low-conf-warning .lcw-body {{
                color: #7A2A24;
                font-size: 0.87rem;
                line-height: 1.45;
            }}

            /* Disclaimer footer */
            .disclaimer {{
                background: linear-gradient(155deg, #FFF6E4 0%, #FFFDF6 100%);
                border: 1px solid #EED9A0;
                border-radius: 14px;
                padding: 1.05rem 1.35rem;
                font-size: 0.85rem;
                color: #7A5C00;
                margin-top: 1.6rem;
                line-height: 1.55;
                box-shadow: 0 1px 0 rgba(255,255,255,0.7) inset, 0 10px 20px rgba(184,114,11,0.08);
            }}

            /* Sidebar — deep panel with an inner glow edge for depth against the canvas */
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #102842 0%, {INK} 55%, #071120 100%);
                box-shadow: 8px 0 30px rgba(0,0,0,0.25);
            }}
            [data-testid="stSidebar"] * {{
                color: #E7EDF3;
            }}
            [data-testid="stSidebar"] > div:first-child {{
                border-right: 1px solid rgba(255,255,255,0.06);
            }}
            .sidebar-title {{
                font-family: 'Fraunces', 'Noto Sans Devanagari', Georgia, serif;
                font-size: 1.3rem;
                font-weight: 600;
                color: #FBF7F0;
                margin-bottom: 0.2rem;
                text-shadow: 0 2px 10px rgba(0,0,0,0.4);
            }}
            .sidebar-sub {{
                font-size: 0.8rem;
                color: #9FB0C2;
                margin-bottom: 1.3rem;
            }}
            .sidebar-label {{
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                color: {CORAL};
                margin: 1.1rem 0 0.6rem 0;
            }}
            .step-item {{
                display: flex;
                align-items: center;
                gap: 0.7rem;
                margin-bottom: 0.7rem;
                font-size: 0.85rem;
                color: #DCE4EC;
                background: rgba(255,255,255,0.035);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 10px;
                padding: 0.55rem 0.7rem;
                box-shadow: 0 4px 10px rgba(0,0,0,0.18);
                transition: transform 0.2s ease, background 0.2s ease;
            }}
            .step-item:hover {{
                transform: translateX(3px);
                background: rgba(255,255,255,0.06);
            }}
            .step-num {{
                background: linear-gradient(155deg, #FF9772 0%, {CORAL} 60%, {CORAL_DARK} 100%);
                color: white;
                width: 22px;
                height: 22px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.7rem;
                font-weight: 700;
                flex-shrink: 0;
                box-shadow: 0 3px 6px rgba(0,0,0,0.35), 0 1px 0 rgba(255,255,255,0.35) inset;
            }}
            [data-testid="stSidebar"] hr {{
                border-color: rgba(255,255,255,0.1);
            }}
            [data-testid="stSidebar"] [data-testid="stExpander"] {{
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
                box-shadow: 0 6px 16px rgba(0,0,0,0.2);
            }}

            /* --------------------------------------------------------------
               Patient record cards get their OWN fixed-color class instead
               of reusing .step-item. .step-item's colors (#DCE4EC text on
               near-transparent background) are tuned for sitting directly
               on the dark navy sidebar. On mobile dark mode, Streamlit's
               native expander can render with a light/white surface, which
               left that near-white text invisible. record-item hardcodes a
               white card + dark ink text that can't be flipped by any
               ambient theme. */
            .record-item {{
                background: #FFFFFF !important;
                border: 1px solid {BORDER};
                border-radius: 10px;
                padding: 0.6rem 0.75rem;
                margin-bottom: 0.7rem;
                box-shadow: 0 4px 10px rgba(0,0,0,0.12);
            }}
            .record-item .rec-name {{
                color: {INK} !important;
                font-weight: 700;
                font-size: 0.88rem;
            }}
            .record-item .rec-id {{
                color: {MUTED} !important;
                font-weight: 400;
            }}
            .record-item .rec-meta {{
                color: {INK_SOFT} !important;
                font-size: 0.78rem;
                opacity: 0.9;
                margin-top: 0.15rem;
            }}

            /* Native buttons app-wide get a soft embossed 3D press */
            .stButton > button, [data-testid="stBaseButton-primary"] {{
                border-radius: 12px !important;
                background: linear-gradient(180deg, #FF8F6B 0%, {CORAL} 55%, {CORAL_DARK} 100%) !important;
                border: none !important;
                color: white !important;
                font-weight: 600 !important;
                box-shadow:
                    0 1px 0 rgba(255,255,255,0.35) inset,
                    0 4px 0 {CORAL_DARK},
                    0 10px 18px rgba(217,84,46,0.35) !important;
                transition: transform 0.12s ease, box-shadow 0.12s ease !important;
            }}
            .stButton > button:hover, [data-testid="stBaseButton-primary"]:hover {{
                transform: translateY(-2px);
                box-shadow:
                    0 1px 0 rgba(255,255,255,0.35) inset,
                    0 6px 0 {CORAL_DARK},
                    0 14px 22px rgba(217,84,46,0.4) !important;
            }}
            .stButton > button:active, [data-testid="stBaseButton-primary"]:active {{
                transform: translateY(3px);
                box-shadow: 0 1px 0 rgba(255,255,255,0.25) inset, 0 1px 0 {CORAL_DARK} !important;
            }}

            /* Progress bars — subtle groove with a glowing fill */
            div[data-testid="stProgress"] > div > div {{
                background: {BORDER} !important;
                border-radius: 8px !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12) inset !important;
            }}
            div[data-testid="stProgress"] > div > div > div {{
                background: linear-gradient(90deg, {TEAL} 0%, #4FD8BE 100%) !important;
                border-radius: 8px !important;
                box-shadow: 0 0 10px rgba(47,191,166,0.5) !important;
            }}

            /* Expanders in the main body get a raised card treatment */
            [data-testid="stExpander"] {{
                border-radius: 14px !important;
                border: 1px solid {BORDER} !important;
                box-shadow: 0 10px 22px rgba(11,30,51,0.08) !important;
                background: {CARD} !important;
            }}

            /* Eye motif drop-shadow already applied inline; keep it grounded on hover */
            .eye-motif-svg {{
                display: block;
            }}

            /* Hide default streamlit chrome we don't need */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}

            /* ------------------------------------------------------------
               FIX: mobile sidebar "»" toggle is unreachable / unclickable.

               Streamlit pins the collapsed-sidebar expand control (the "»"
               chevron) to the very top-left corner of the viewport. On a
               phone's mobile browser that spot sits underneath the OS
               status bar / notch (clock, wifi, notification icons), so the
               button is either partly hidden behind that bar or too close
               to the screen edge for a touch to register. Push it down
               below the safe area, give it a real square tap target, and
               raise its stacking order above everything else so it's
               always reachable, on every screen size.
               ------------------------------------------------------------ */
            [data-testid="stSidebarCollapsedControl"] {{
                position: fixed !important;
                top: calc(env(safe-area-inset-top, 0px) + 14px) !important;
                left: 14px !important;
                z-index: 999999 !important;
                background: {CARD} !important;
                border: 1px solid {BORDER} !important;
                border-radius: 12px !important;
                box-shadow: 0 6px 16px rgba(11,30,51,0.18) !important;
                width: 56px !important;
                height: 56px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                touch-action: manipulation !important;
                pointer-events: auto !important;
            }}
            [data-testid="stSidebarCollapsedControl"] button {{
                width: 100% !important;
                height: 100% !important;
                pointer-events: auto !important;
            }}
            [data-testid="stSidebarCollapsedControl"] svg {{
                width: 30px !important;
                height: 30px !important;
            }}
            /* The header bar Streamlit draws behind that control can also
               intercept taps on mobile — make sure it never blocks clicks
               on anything sitting above it. */
            [data-testid="stHeader"] {{
                pointer-events: none !important;
                background: transparent !important;
            }}
            [data-testid="stHeader"] * {{
                pointer-events: auto !important;
            }}
            @media (max-width: 700px) {{
                [data-testid="stSidebarCollapsedControl"] {{
                    top: calc(env(safe-area-inset-top, 0px) + 10px) !important;
                    left: 10px !important;
                    width: 60px !important;
                    height: 60px !important;
                }}
                [data-testid="stSidebarCollapsedControl"] svg {{
                    width: 32px !important;
                    height: 32px !important;
                }}
            }}

            /* ------------------------------------------------------------
               Side rails — the empty margins on wide screens (either side
               of the centered 1080px content) get a few floating chips
               tied to the problem/solution, plus a faint vein-line motif,
               instead of sitting bare. They're fixed/pointer-events:none
               so they never interfere with the real UI, and they hide
               themselves on narrower windows where there's no room.
               ------------------------------------------------------------ */
            .side-rail {{
                position: fixed;
                top: 12%;
                width: 190px;
                z-index: 0;
                pointer-events: none;
                display: flex;
                flex-direction: column;
                gap: 2.2rem;
            }}
            .side-rail-left  {{ left: 300px; }}
            .side-rail-right {{ right: 32px; }}
            @media (max-width: 1650px) {{
                .side-rail {{ display: none; }}
            }}
            .rail-vein {{
                position: absolute;
                top: -60px;
                bottom: -60px;
                left: 50%;
                width: 40px;
                transform: translateX(-50%);
                z-index: -1;
                opacity: 0.5;
            }}
            .rail-chip {{
                background: rgba(255,255,255,0.72);
                backdrop-filter: blur(6px);
                border: 1px solid {BORDER};
                border-radius: 16px;
                padding: 0.85rem 1rem;
                box-shadow: 0 10px 24px rgba(11,30,51,0.08);
                font-size: 0.76rem;
                line-height: 1.4;
                color: {INK_SOFT};
                animation: floatY 6.5s ease-in-out infinite;
            }}
            .rail-chip .rail-icon {{
                font-size: 1.25rem;
                display: block;
                margin-bottom: 0.3rem;
            }}
            .side-rail .rail-chip:nth-child(2) {{ animation-delay: 1.3s; }}
            .side-rail .rail-chip:nth-child(3) {{ animation-delay: 2.6s; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Decorative side rails — fills the wide-screen margins either side of the
# centered content with a few chips tied to the problem/solution (why this
# tool exists, what makes it different), plus a faint vein-line motif that
# echoes the retina/optic-nerve art used elsewhere in the app.
# ----------------------------------------------------------------------------
def render_side_rails():
    left_items = [
        ("🩺", "Early screening prevents avoidable vision loss"),
        ("📍", "Built for areas with no eye specialist nearby"),
        ("🧑‍⚕️", "Designed for ASHA & community health workers"),
    ]
    right_items = [
        ("🔍", "Explainable AI — see exactly what it looked at"),
        ("⚡", "A screening result in seconds, not weeks"),
        ("🌱", "Smart India Hackathon 2026 · Clean &amp; Green Tech"),
    ]

    vein_svg = """
    <svg class="rail-vein" viewBox="0 0 40 400" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M20 0 C30 40 8 70 20 110 C32 150 8 180 20 220 C30 255 10 285 20 320 C28 350 12 375 20 400"
            fill="none" stroke="#2FBFA6" stroke-width="2" stroke-linecap="round"/>
      <path d="M20 60 C14 75 4 78 -2 90" fill="none" stroke="#FF7A50" stroke-width="1.4" opacity="0.6" stroke-linecap="round"/>
      <path d="M20 180 C26 195 36 198 42 210" fill="none" stroke="#FF7A50" stroke-width="1.4" opacity="0.6" stroke-linecap="round"/>
      <path d="M20 300 C14 315 4 318 -2 330" fill="none" stroke="#FF7A50" stroke-width="1.4" opacity="0.6" stroke-linecap="round"/>
    </svg>
    """

    def rail_html(items):
        return "".join(
            f'<div class="rail-chip"><span class="rail-icon">{icon}</span>{text}</div>'
            for icon, text in items
        )

    st.markdown(
        f"""
        <div class="side-rail side-rail-left">{vein_svg}{rail_html(left_items)}</div>
        <div class="side-rail side-rail-right">{vein_svg}{rail_html(right_items)}</div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Live 3D tilt — tracks the mouse and rotates any [data-tilt] element in real
# perspective space (not just a CSS hover state), which is what makes the
# hero card / result card / upload card feel like physical objects instead
# of flat rectangles. Runs from a components.html iframe since Streamlit
# strips <script> tags from st.markdown.
# ----------------------------------------------------------------------------
def inject_tilt_script():
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            function attach(el) {
                if (el.dataset.tiltReady) return;
                el.dataset.tiltReady = "1";
                const strength = parseFloat(el.getAttribute('data-tilt')) || 6;
                el.addEventListener('mousemove', (e) => {
                    const r = el.getBoundingClientRect();
                    const px = (e.clientX - r.left) / r.width - 0.5;
                    const py = (e.clientY - r.top) / r.height - 0.5;
                    el.style.transform =
                        `perspective(1000px) rotateX(${(-py * strength).toFixed(2)}deg) ` +
                        `rotateY(${(px * strength).toFixed(2)}deg) translateZ(4px)`;
                });
                el.addEventListener('mouseleave', () => {
                    el.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateZ(0px)';
                });
            }
            function scan() {
                doc.querySelectorAll('[data-tilt]').forEach(attach);
            }
            scan();
            new MutationObserver(scan).observe(doc.body, {childList: true, subtree: true});
        })();
        </script>
        """,
        height=0,
        width=0,
    )


# ----------------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_metadata():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH)):
        return None, None

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, len(metadata["class_names"])),
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


# ----------------------------------------------------------------------------
# Patient records — simple local CSV log so a health worker can save a
# patient's ID/name with each screening and look up their history later.
# Note: grade/severity_label are stored in English (the model's own class
# names), independent of UI language, so historical records stay consistent
# even if the language is switched later. They're re-labelled via t() only
# when displayed in the sidebar list below.
# ----------------------------------------------------------------------------
def save_patient_record(patient_id, patient_name, pred_class, severity_label, confidence):
    file_exists = os.path.exists(RECORDS_PATH)
    with open(RECORDS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORDS_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_id": patient_id.strip(),
            "patient_name": patient_name.strip(),
            "grade": pred_class,
            "severity_label": severity_label,
            "confidence": f"{confidence:.1f}",
        })


def load_patient_records(search=""):
    if not os.path.exists(RECORDS_PATH):
        return []
    with open(RECORDS_PATH, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if search:
        search_lower = search.strip().lower()
        rows = [
            r for r in rows
            if search_lower in r.get("patient_id", "").lower()
            or search_lower in r.get("patient_name", "").lower()
        ]
    return list(reversed(rows))  # most recent first


# ----------------------------------------------------------------------------
# PDF report generation — a clean, printable summary of the screening result
# that a health worker can download and hand to the patient or ophthalmologist.
#
# NOTE: this stays in English regardless of the UI language. FPDF's built-in
# core fonts (Helvetica etc.) don't support Devanagari or other Indic
# scripts, and referral reports going to a specialist are conventionally in
# English in India anyway. See translations.py's docstring if you want to
# add a Hindi PDF via an embedded Unicode font.
# ----------------------------------------------------------------------------
def _np_img_to_bytes(img_array):
    """RGB numpy array -> PNG bytes, for embedding into the PDF."""
    buf = io.BytesIO()
    Image.fromarray(img_array.astype("uint8")).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _pdf_safe(text) -> str:
    """
    Make text safe for FPDF's core fonts (Helvetica etc.), which only
    support Latin-1. Without this, any em dash, curly quote, emoji, or
    non-Latin script (e.g. a patient name typed in Hindi) raises
    FPDFUnicodeEncodingException and crashes report generation.
    """
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "\u2014": "-", "\u2013": "-",   # em dash, en dash
        "\u2018": "'", "\u2019": "'",   # curly single quotes
        "\u201c": '"', "\u201d": '"',   # curly double quotes
        "\u2026": "...",                 # ellipsis
        "\u2022": "-",                    # bullet
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Last-resort safety net: silently drop anything still outside Latin-1
    # instead of crashing the whole report.
    return text.encode("latin-1", "ignore").decode("latin-1")


def generate_pdf_report(
    patient_id,
    patient_name,
    pred_class,
    severity_label,
    confidence,
    style,
    recommendation,
    processed_img,
    overlay_img,
):
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # --- Header -----------------------------------------------------------
    pdf.set_fill_color(11, 30, 51)  # INK
    pdf.rect(0, 0, 210, 28, style="F")
    pdf.set_xy(12, 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 8, "Drishti - DR Screening Report", ln=True)
    pdf.set_x(12)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(220, 228, 236)
    pdf.cell(0, 6, "Explainable AI Screening for Diabetic Retinopathy", ln=True)

    pdf.set_text_color(20, 20, 20)
    pdf.ln(14)

    # --- Patient details ----------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Patient Details", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, _pdf_safe(f"Patient Name: {patient_name.strip() or '-'}"), ln=True)
    pdf.cell(0, 7, _pdf_safe(f"Patient ID: {patient_id.strip() or '-'}"), ln=True)
    pdf.cell(0, 7, _pdf_safe(f"Report generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}"), ln=True)
    pdf.ln(4)

    # --- Result summary -----------------------------------------------------
    sev_rgb = tuple(int(style["color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Screening Result", ln=True)
    pdf.set_fill_color(*sev_rgb)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, _pdf_safe(f"  {style['label']} - Grade {pred_class}: {severity_label}"), ln=True, fill=True)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, _pdf_safe(f"Model confidence: {confidence:.1f}%"), ln=True)
    pdf.ln(2)

    # --- Images side by side -------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Retinal Images", ln=True)
    img_w = 85
    y_before = pdf.get_y()
    pdf.image(_np_img_to_bytes(processed_img), x=12, y=y_before, w=img_w)
    pdf.image(_np_img_to_bytes(overlay_img), x=12 + img_w + 6, y=y_before, w=img_w)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_y(y_before + img_w * (processed_img.shape[0] / processed_img.shape[1]) + 2)
    pdf.cell(img_w, 5, "Uploaded image (preprocessed)", align="C")
    pdf.cell(6, 5, "")
    pdf.cell(img_w, 5, "Grad-CAM - AI attention map", align="C", ln=True)
    pdf.ln(6)

    # --- Recommendation -------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommended Action", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6.5, _pdf_safe(recommendation))
    pdf.ln(4)

    # --- Disclaimer -------------------------------------------------------
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 90, 20)
    pdf.multi_cell(
        0, 5,
        _pdf_safe(
            "Disclaimer: This is an AI-assisted screening tool, intended to support - not replace - "
            "clinical judgment. Always have a qualified ophthalmologist review this result before any "
            "treatment decisions."
        )
    )

    return bytes(pdf.output(dest="S"))


@st.dialog(t("report_dialog_header"))
def show_report_download_dialog():
    st.write(t("report_dialog_body"))
    st.download_button(
        label=t("download_report_button"),
        data=st.session_state["report_pdf_bytes"],
        file_name=st.session_state["report_filename"],
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )
    if st.button(t("close_button"), use_container_width=True):
        st.rerun()


# ----------------------------------------------------------------------------
# Preprocessing (must mirror training-time preprocessing exactly)
# ----------------------------------------------------------------------------
def crop_black_border(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def ben_graham_preprocess(img, sigma_frac=10):
    sigma = img.shape[1] / sigma_frac
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def preprocess_image(pil_img, size):
    img = np.array(pil_img.convert("RGB"))
    img = crop_black_border(img)
    img = cv2.resize(img, (size, size))
    img = ben_graham_preprocess(img)
    return img


def is_likely_fundus_image(pil_img, min_warm_frac=0.35, aspect_tol=0.35):
    """Heuristic sanity check: does this look like a retinal fundus photo?

    This is NOT a trained classifier — it's a fast, rule-based filter to
    catch obviously wrong uploads (selfies, screenshots, documents, random
    photos) before they reach the DR model. It looks for two things real
    fundus photos reliably have:

      1. A warm, reddish-orange color cast — the retina, optic disc, and
         vasculature give fundus photos a very distinctive palette that
         most everyday photos don't share.
      2. A roughly circular/square capture region — fundus cameras crop to
         a circle (often on a black background), so after removing any
         black border the remaining image is close to square, not a long
         thin rectangle like a typical document or landscape photo.

    Returns (looks_like_fundus: bool, aspect_ratio: float, warm_frac: float)
    so the caller can log/display the raw scores if useful.
    """
    img = np.array(pil_img.convert("RGB"))
    cropped = crop_black_border(img)

    h, w = cropped.shape[:2]
    aspect_ratio = min(h, w) / max(h, w) if max(h, w) > 0 else 0
    is_squarish = aspect_ratio >= (1 - aspect_tol)

    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    # Red/orange hues wrap around 0 on OpenCV's 0-179 hue scale.
    warm_mask = ((hue <= 25) | (hue >= 165)) & (sat > 40)
    warm_frac = float(warm_mask.mean())
    is_warm_toned = warm_frac >= min_warm_frac

    return (is_squarish and is_warm_toned), aspect_ratio, warm_frac


def compute_blur_score(pil_img):
    """Variance of the Laplacian — a standard, lightweight focus measure.

    A sharp, in-focus image has lots of high-frequency detail (crisp edges,
    vessel borders, etc.), which the Laplacian responds to strongly, giving
    a high variance. A blurry image has smoothed-out edges, so the variance
    is low. This runs in milliseconds and needs no extra model.
    """
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        probs = F.softmax(output, dim=1)[0].detach().numpy()
        return cam.numpy(), class_idx, probs


def overlay_heatmap(orig_img, cam, alpha=0.42):
    cam_resized = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_img, 1 - alpha, heatmap, alpha, 0)
    return overlay


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        # Language selector goes first — everything else in the sidebar (and
        # the rest of the page, on the next rerun) reads t() using whatever
        # is chosen here.
        lang_codes = list(LANGUAGES.keys())
        selected_lang = st.selectbox(
            t("language_label"),
            options=lang_codes,
            format_func=lambda code: LANGUAGES[code],
            index=lang_codes.index(st.session_state.lang),
            key="lang_selectbox",
        )
        if selected_lang != st.session_state.lang:
            st.session_state.lang = selected_lang
            st.rerun()

        st.markdown(f'<div class="sidebar-title">{t("app_title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-sub">{t("app_subtitle")}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="sidebar-label">{t("how_it_works")}</div>', unsafe_allow_html=True)
        steps = [t("step_1"), t("step_2"), t("step_3"), t("step_4"), t("step_5")]
        for i, s in enumerate(steps, 1):
            st.markdown(
                f'<div class="step-item"><div class="step-num">{i}</div><div>{s}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(f'<div class="sidebar-label">{t("severity_scale")}</div>', unsafe_allow_html=True)
        for idx, style in SEVERITY_STYLE.items():
            grade_name = t(f"grade_{idx}")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;font-size:0.85rem;">'
                f'<span style="width:10px;height:10px;border-radius:50%;background:{style["color"]};display:inline-block;"></span>'
                f'<span>{t("grade_word")} {idx} — {grade_name}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(f'<div class="sidebar-label">{t("eye_anatomy_ref")}</div>', unsafe_allow_html=True)
        if os.path.exists(EYE_DIAGRAM_PATH):
            with st.expander(t("view_labeled_diagram")):
                st.image(EYE_DIAGRAM_PATH, use_container_width=True)
                st.caption(t("eye_anatomy_caption"))

        st.markdown("---")
        st.markdown(f'<div class="sidebar-label">{t("records_label")}</div>', unsafe_allow_html=True)
        search_query = st.text_input(
            t("records_search_placeholder"),
            key="record_search",
            placeholder=t("records_search_placeholder"),
            label_visibility="collapsed",
        )
        records = load_patient_records(search_query)
        if records:
            with st.expander(t("records_expander_label").format(n=len(records)), expanded=bool(search_query)):
                for r in records[:20]:
                    patient_display_name = r["patient_name"] or "—"
                    id_display = r["patient_id"] or t("no_id_label")
                    st.markdown(
                        f'<div class="record-item">'
                        f'<div class="rec-name">{patient_display_name} '
                        f'<span class="rec-id">({id_display})</span></div>'
                        f'<div class="rec-meta">{r["timestamp"]} · {t("grade_word")} {r["grade"]} — '
                        f'{r["severity_label"]} · {r["confidence"]}%</div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption(t("no_records_yet") if not search_query else t("no_matches_found"))

        st.markdown("---")
        st.caption(t("sidebar_footer"))


# ----------------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------------
inject_css()
inject_tilt_script()
render_sidebar()
render_side_rails()

st.markdown(
    f"""
    <div class="hero" data-tilt="6">
        <div class="glass-orb orb1"></div>
        <div class="glass-orb orb2"></div>
        <div class="glass-orb orb3"></div>
        <div class="hero-inner">
            <div class="hero-copy">
                <div class="hero-tag">{t("hero_tag")}</div>
                <h1>{t("hero_title")}</h1>
                <p>{t("hero_body")}</p>
            </div>
            <div class="hero-eye">{EYE_MOTIF_SVG}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model, metadata = load_model_and_metadata()

if model is None:
    st.error(t("model_missing_error"))
    st.stop()

class_names = {int(k): v for k, v in metadata["class_names"].items()}
img_size = metadata["img_size"]
mean, std = metadata["mean"], metadata["std"]

infer_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

st.markdown(f'<div class="section-label">{t("patient_details_label")}</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-card" data-tilt="4" style="padding:1.1rem 1.3rem;">', unsafe_allow_html=True)
pcol1, pcol2 = st.columns(2)
with pcol1:
    patient_id = st.text_input(t("patient_id_label"), key="patient_id", placeholder=t("patient_id_placeholder"))
with pcol2:
    patient_name = st.text_input(t("patient_name_label"), key="patient_name", placeholder=t("patient_name_placeholder"))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="section-label">{t("upload_section_label")}</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-card" data-tilt="4">', unsafe_allow_html=True)

tab_file, tab_camera = st.tabs([f"📁 {t('upload_mode_file')}", f"📷 {t('upload_mode_camera')}"])
with tab_file:
    file_upload = st.file_uploader(
        t("upload_prompt"),
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
with tab_camera:
    camera_capture = st.camera_input(
        t("camera_prompt"),
        label_visibility="collapsed",
    )

# Camera takes priority when both are present (i.e. the health worker just
# captured a fresh photo after having an older file already selected).
uploaded_file = camera_capture if camera_capture is not None else file_upload
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None and not patient_id.strip():
    st.markdown(
        f"""
        <div class="low-conf-warning">
            <div>🪪</div>
            <div>
                <div class="lcw-title">{t("patient_id_required_title")}</div>
                <div class="lcw-body">{t("patient_id_required_body")}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    looks_like_fundus, _aspect_ratio, _warm_frac = is_likely_fundus_image(pil_img)
    if not looks_like_fundus:
        st.markdown(
            f"""
            <div class="low-conf-warning">
                <div>🚫</div>
                <div>
                    <div class="lcw-title">{t("not_fundus_title")}</div>
                    <div class="lcw-body">{t("not_fundus_body")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    blur_score = compute_blur_score(pil_img)

    if blur_score < BLUR_THRESHOLD:
        st.markdown(
            f"""
            <div class="low-conf-warning">
                <div>🔍</div>
                <div>
                    <div class="lcw-title">{t("blurry_title")}</div>
                    <div class="lcw-body">{t("blurry_body")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    with st.spinner(t("analyzing_spinner")):
        processed = preprocess_image(pil_img, img_size)
        tensor = infer_tfms(processed).unsqueeze(0)
        tensor.requires_grad_(True)

        cam, pred_class, probs = gradcam.generate(tensor)
        overlay = overlay_heatmap(processed, cam)

    severity_label = class_names[pred_class]
    confidence = probs[pred_class] * 100
    style = SEVERITY_STYLE[pred_class]
    style_label = t(style["label_key"])
    recommendation_text = t(f"rec_{pred_class}")
    # The PDF report always renders in English (see generate_pdf_report's
    # docstring) since FPDF's core fonts can't render Devanagari/Indic
    # scripts -- build a separate English-only copy for it regardless of
    # the current UI language.
    recommendation_text_pdf = t(f"rec_{pred_class}", lang_override="en")

    save_patient_record(patient_id, patient_name, pred_class, severity_label, confidence)

    # Build the downloadable PDF report and pop up a "download it" dialog once
    # per new result (guarded by a signature so re-running the script on
    # unrelated widget interactions doesn't reopen the popup every time).
    report_pdf_bytes = generate_pdf_report(
        patient_id, patient_name, pred_class, severity_label, confidence,
        {**style, "label": style_label}, recommendation_text_pdf, processed, overlay,
    )
    safe_id = (patient_id.strip() or "patient").replace(" ", "_")
    st.session_state["report_pdf_bytes"] = report_pdf_bytes
    st.session_state["report_filename"] = f"Drishti_Report_{safe_id}.pdf"

    result_signature = f"{uploaded_file.name}-{uploaded_file.size}-{patient_id.strip()}"
    if st.session_state.get("report_popup_shown_for") != result_signature:
        st.session_state["report_popup_shown_for"] = result_signature
        show_report_download_dialog()

    st.markdown(
        f'<div class="section-label">{t("patient_section_label")}</div>'
        f'<div style="margin:-0.4rem 0 1rem 0; font-size:0.95rem; color:{INK_SOFT};">'
        f'🪪 <strong>{patient_name.strip() or "—"}</strong> &nbsp;·&nbsp; ID: <strong>{patient_id.strip()}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(processed, use_container_width=True)
        st.markdown(f'<div class="img-caption">{t("original_image_caption")}</div>', unsafe_allow_html=True)
    with col2:
        st.image(overlay, use_container_width=True)
        st.markdown(f'<div class="img-caption">{t("heatmap_caption")}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-card" data-tilt="7" style="--sev-color: {style['color']}; --sev-bg: {style['bg']};">
            <p class="result-sub">{style['icon']} {style_label} · {t("grade_word")} {pred_class}</p>
            <p class="result-title">{severity_label}</p>
            <span class="confidence-pill">{t("confidence_label")}: {confidence:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        # Show the second-most-likely class too, so the reviewer sees the ambiguity
        sorted_idx = np.argsort(probs)[::-1]
        second_idx = int(sorted_idx[1])
        st.markdown(
            f"""
            <div class="low-conf-warning">
                <div>⚠️</div>
                <div>
                    <div class="lcw-title">{t("low_conf_review_title")}</div>
                    <div class="lcw-body">{t("low_conf_review_body").format(
                        confidence=f"{confidence:.1f}",
                        grade=f"<strong>{class_names[second_idx]}</strong>",
                        prob=f"{probs[second_idx]*100:.1f}",
                    )}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="section-label">{t("recommended_action_label")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rec-box">💡 {recommendation_text}</div>', unsafe_allow_html=True)

    st.download_button(
        label=t("download_report_button"),
        data=st.session_state["report_pdf_bytes"],
        file_name=st.session_state["report_filename"],
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )

    with st.expander(t("confidence_breakdown_label")):
        for cls_idx in sorted(class_names.keys()):
            grade_label = t(f"grade_{cls_idx}")
            st.progress(float(probs[cls_idx]), text=f"{grade_label}: {probs[cls_idx]*100:.1f}%")

    st.markdown(
        f"""
        <div class="disclaimer">
            {t("disclaimer")}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 3.2rem 1rem; background: linear-gradient(155deg, #FFFFFF 0%, #FBF6EE 100%);
                    border-radius: 20px; border: 1px solid {BORDER};
                    box-shadow: 0 1px 0 rgba(255,255,255,0.8) inset, 0 16px 30px rgba(11,30,51,0.07);">
            <div style="display:flex; justify-content:center; margin-bottom: 0.8rem;">
                <div style="animation: floatY 5s ease-in-out infinite; filter: drop-shadow(0 12px 16px rgba(11,30,51,0.18));">
                    {EYE_MOTIF_SVG.replace('width="240" height="240"', 'width="96" height="96"')}
                </div>
            </div>
            <div style="font-family:'Fraunces', 'Noto Sans Devanagari', Georgia, serif; font-weight:600; font-size: 1.15rem; color: {INK};">{t("no_image_title")}</div>
            <div style="font-size: 0.9rem; margin-top: 0.3rem; color: {MUTED};">{t("no_image_body")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
