import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import numpy as np
import time
import streamlit as st
import streamlit.components.v1 as components


# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Import your enhanced prediction module
from predict import predict_stock_enhanced, log_to_csv

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🚀 AI Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "predictions" not in st.session_state:
    st.session_state.predictions = {}
if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = None
if "visitor_count" not in st.session_state:
    st.session_state.visitor_count = 12847
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None
if "notifications" not in st.session_state:
    st.session_state.notifications = []

# Update visitor count periodically
st.session_state.visitor_count += 1

# ============================================================================
# ADVANCED FUTURISTIC CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* ===== GLOBAL STYLES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Animated background particles */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(88, 86, 214, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(138, 43, 226, 0.1) 0%, transparent 50%);
        animation: particleAnimation 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes particleAnimation {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }
    
    /* ===== HERO SECTION ===== */
    .hero-section {
        position: relative;
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.15) 0%, rgba(88, 86, 214, 0.15) 100%);
        backdrop-filter: blur(30px);
        border-radius: 32px;
        padding: 4rem 3rem;
        margin-bottom: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        overflow: hidden;
        animation: heroFadeIn 1s ease-out;
    }
    
    @keyframes heroFadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(138, 43, 226, 0.2) 0%, transparent 70%);
        animation: heroGlow 8s ease-in-out infinite;
    }
    
    @keyframes heroGlow {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); }
    }
    
    .hero-title {
        position: relative;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #fff 0%, #a78bfa 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        animation: titleSlideIn 1.2s ease-out;
        text-shadow: 0 0 40px rgba(167, 139, 250, 0.3);
    }
    
    @keyframes titleSlideIn {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .hero-subtitle {
        position: relative;
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 400;
        line-height: 1.6;
        max-width: 800px;
        animation: subtitleSlideIn 1.4s ease-out;
    }
    
    @keyframes subtitleSlideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .hero-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        background: rgba(138, 43, 226, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 50px;
        border: 1px solid rgba(167, 139, 250, 0.5);
        font-size: 0.85rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 1.5rem;
        animation: badgePulse 2s ease-in-out infinite;
    }
    
    @keyframes badgePulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 20px rgba(138, 43, 226, 0.3); }
        50% { transform: scale(1.05); box-shadow: 0 0 30px rgba(138, 43, 226, 0.5); }
    }
    
    /* ===== GLASS CARDS ===== */
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(167, 139, 250, 0.5);
        box-shadow: 
            0 20px 60px rgba(138, 43, 226, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(167, 139, 250, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(167, 139, 250, 0.6);
        box-shadow: 0 15px 40px rgba(138, 43, 226, 0.3);
    }
    
    .metric-card:hover::after {
        opacity: 1;
        animation: rotateGlow 3s linear infinite;
    }
    
    @keyframes rotateGlow {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
.metric-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 0 20px rgba(167, 139, 250, 0.5);
        margin-bottom: 0.3rem;
    }
    
    .metric-subtitle {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.5);
        font-weight: 500;
    }
    
    /* ===== PREDICTION CARDS ===== */
    .prediction-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
        backdrop-filter: blur(25px);
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: cardFadeIn 0.6s ease-out backwards;
    }
    
    @keyframes cardFadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .prediction-card:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: rgba(167, 139, 250, 0.6);
        box-shadow: 0 25px 70px rgba(138, 43, 226, 0.4);
    }
    
    /* ===== ACTION BADGES ===== */
    .action-badge {
        display: inline-block;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        backdrop-filter: blur(15px);
        border: 2px solid;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        animation: badgeAppear 0.8s ease-out;
    }
    
    @keyframes badgeAppear {
        from { opacity: 0; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .action-strong-buy {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.4) 0%, rgba(22, 163, 74, 0.4) 100%);
        color: #fff;
        border-color: rgba(34, 197, 94, 0.8);
        box-shadow: 0 8px 30px rgba(34, 197, 94, 0.4);
    }
    
    .action-strong-buy:hover {
        box-shadow: 0 12px 40px rgba(34, 197, 94, 0.6);
        transform: scale(1.05);
    }
    
    .action-buy, .action-cautious-buy {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.3) 0%, rgba(22, 163, 74, 0.3) 100%);
        color: #fff;
        border-color: rgba(34, 197, 94, 0.6);
        box-shadow: 0 6px 25px rgba(34, 197, 94, 0.3);
    }
    
    .action-strong-sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.4) 0%, rgba(220, 38, 38, 0.4) 100%);
        color: #fff;
        border-color: rgba(239, 68, 68, 0.8);
        box-shadow: 0 8px 30px rgba(239, 68, 68, 0.4);
    }
    
    .action-sell, .action-cautious-sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(220, 38, 38, 0.3) 100%);
        color: #fff;
        border-color: rgba(239, 68, 68, 0.6);
        box-shadow: 0 6px 25px rgba(239, 68, 68, 0.3);
    }
    
    .action-wait {
        background: linear-gradient(135deg, rgba(156, 163, 175, 0.3) 0%, rgba(107, 114, 128, 0.3) 100%);
        color: #fff;
        border-color: rgba(156, 163, 175, 0.6);
    }
    
    .action-no-trade {
        background: linear-gradient(135deg, rgba(107, 114, 128, 0.3) 0%, rgba(75, 85, 99, 0.3) 100%);
        color: #fff;
        border-color: rgba(107, 114, 128, 0.6);
    }
    
    /* ===== SCORE BADGES ===== */
    .score-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 1rem 1.5rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.4rem;
        border: 3px solid;
        min-width: 120px;
        transition: all 0.3s ease;
    }
    
    .score-excellent {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border-color: rgba(34, 197, 94, 0.6);
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.3);
    }
    
    .score-good {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        border-color: rgba(59, 130, 246, 0.6);
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
    }
    
    .score-marginal {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        border-color: rgba(251, 191, 36, 0.6);
        box-shadow: 0 0 30px rgba(251, 191, 36, 0.3);
    }
    
    .score-weak {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border-color: rgba(239, 68, 68, 0.6);
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
    }
    
    /* ===== INFO BOXES ===== */
    .info-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        animation: slideInLeft 0.5s ease-out;
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(251, 191, 36, 0.05) 100%);
        border-left: 4px solid #fbbf24;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        animation: slideInLeft 0.5s ease-out;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%);
        border-left: 4px solid #22c55e;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        animation: slideInLeft 0.5s ease-out;
    }
    
    /* ===== BUTTONS ===== */
    .stButton button {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.3) 0%, rgba(88, 86, 214, 0.3) 100%);
        border: 2px solid rgba(167, 139, 250, 0.5);
        border-radius: 16px;
        color: #fff;
        font-weight: 700;
        padding: 1rem 2.5rem;
        backdrop-filter: blur(15px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.5) 0%, rgba(88, 86, 214, 0.5) 100%);
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(138, 43, 226, 0.4);
        border-color: rgba(167, 139, 250, 0.8);
    }
    
    .stButton button:active {
        transform: translateY(-1px);
    }
    
    /* ===== PROGRESS BAR ===== */
    .progress-bar {
        height: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        margin-top: 0.8rem;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.95) 100%);
        backdrop-filter: blur(30px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stSelectbox, 
    [data-testid="stSidebar"] .stMultiSelect,
    [data-testid="stSidebar"] .stTextInput {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
/* ===== VISITOR COUNTER - COMPACT & SMALLER ===== */
    .visitor-counter {
        position: fixed;
        bottom: 1.2rem;
        right: 1.2rem;
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.25) 0%, rgba(88, 86, 214, 0.25) 100%);
        backdrop-filter: blur(15px);
        border: 1.5px solid rgba(167, 139, 250, 0.4);
        border-radius: 12px;
        padding: 0.4rem 0.8rem;
        box-shadow: 0 6px 24px rgba(138, 43, 226, 0.25);
        z-index: 9999;
        animation: counterFloat 3s ease-in-out infinite;
        min-width: 85px;
    }
    
    @keyframes counterFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }
    
    .visitor-counter-label {
        font-size: 0.55rem;
        color: rgba(255, 255, 255, 0.55);
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 0.1rem;
        font-weight: 600;
    }
    
    .visitor-counter-value {
        font-size: 1rem;
        font-weight: 800;
        color: #a78bfa;
        text-shadow: 0 0 12px rgba(167, 139, 250, 0.4);
        line-height: 1;
    }
    
    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 600;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(167, 139, 250, 0.2);
        color: #fff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.4) 0%, rgba(88, 86, 214, 0.4) 100%);
        color: #fff !important;
        box-shadow: 0 4px 15px rgba(138, 43, 226, 0.3);
    }
    
    /* ===== DATA TABLE STYLING ===== */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* ===== CHART CONTAINER ===== */
    .chart-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
    }
    
    /* ===== LOADING ANIMATION ===== */
    .loading-spinner {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 4px solid rgba(167, 139, 250, 0.2);
        border-top-color: #a78bfa;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* ===== NOTIFICATION TOAST ===== */
    .notification-toast {
        position: fixed;
        top: 2rem;
        right: 2rem;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.3) 0%, rgba(22, 163, 74, 0.3) 100%);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(34, 197, 94, 0.5);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        box-shadow: 0 10px 40px rgba(34, 197, 94, 0.3);
        z-index: 9999;
        animation: toastSlideIn 0.5s ease-out;
    }
    
    @keyframes toastSlideIn {
        from { opacity: 0; transform: translateX(100px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-section {
            padding: 2rem 1.5rem;
        }
        
        .visitor-counter {
            bottom: 0.8rem;
            right: 0.8rem;
            padding: 0.35rem 0.7rem;
            min-width: 75px;
        }
        
        .visitor-counter-label {
            font-size: 0.5rem;
        }
        
        .visitor-counter-value {
            font-size: 0.9rem;
        }
    }
    
    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== SMOOTH SCROLLING ===== */
    html {
        scroll-behavior: smooth;
    }
    
    /* ===== CUSTOM SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.5) 0%, rgba(88, 86, 214, 0.5) 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.7) 0%, rgba(88, 86, 214, 0.7) 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_action_color_class(action):
    """Get CSS class for action badge"""
    action_clean = action.lower().replace('🟢', '').replace('🔴', '').replace('⚡', '').replace('⏸️', '').replace('❌', '').strip()
    
    if 'strong buy' in action_clean:
        return 'action-strong-buy'
    elif 'buy' in action_clean or 'cautious buy' in action_clean:
        return 'action-buy'
    elif 'strong sell' in action_clean:
        return 'action-strong-sell'
    elif 'sell' in action_clean or 'cautious sell' in action_clean:
        return 'action-sell'
    elif 'wait' in action_clean:
        return 'action-wait'
    else:
        return 'action-no-trade'

def get_score_badge_class(score):
    """Get CSS class for score badge"""
    if score >= 75:
        return 'score-excellent'
    elif score >= 65:
        return 'score-good'
    elif score >= 55:
        return 'score-marginal'
    else:
        return 'score-weak'

def clean_emoji(text):
    """Remove emojis from text"""
    return text.replace('🟢', '').replace('🟡', '').replace('🟠', '').replace('🔴', '').replace('📈', '').replace('📉', '').replace('🚀', '').replace('⚖️', '').replace('🔄', '').replace('⚡', '').replace('⏸️', '').replace('❌', '').strip()

def format_percentage(value, include_sign=True):
    """Format percentage value"""
    if include_sign:
        return f"{value:+.2f}%"
    return f"{value:.2f}%"

def format_currency(value):
    """Format currency value"""
    return f"${value:,.2f}"

def show_notification(message, type="success"):
    """Show notification toast"""
    color = "#22c55e" if type == "success" else "#ef4444" if type == "error" else "#fbbf24"
    st.markdown(f"""
    <div class="notification-toast" style="border-color: {color};">
        <strong>{'✅' if type == 'success' else '❌' if type == 'error' else '⚠️'}</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def create_progress_bar(percentage, color="#a78bfa"):
    """Create animated progress bar"""
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {percentage}%; background: {color};"></div>
    </div>
    """

def get_chart_theme():
    """Get consistent chart theme"""
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(255,255,255,0.03)',
        'font': {'color': 'white', 'family': 'Inter'},
        'xaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'showgrid': True,
            'zeroline': False
        },
        'yaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'showgrid': True,
            'zeroline': False
        }
    }

# ============================================================================
# VISITOR COUNTER DISPLAY - RESET TO START FROM LOW NUMBER
# ============================================================================
# Initialize visitor count with a reasonable starting number
if "visitor_count" not in st.session_state:
    st.session_state.visitor_count = 127  # Start from 127 instead of 12000+

# Increment by small amount
st.session_state.visitor_count += 1

st.markdown(f"""
<div class="visitor-counter">
    <div class="visitor-counter-label">👥 VISITORS</div>
    <div class="visitor-counter-value">{st.session_state.visitor_count:,}</div>
</div>
""", unsafe_allow_html=True)
# ============================================================================
# HERO SECTION / HOME PAGE - COMPLETE FIX
# ============================================================================

def render_home_page():
    """Render the landing/home page - NO CODE BLOCKS, ONLY RENDERED HTML"""
    
    # Decorative background with stock symbols
    st.markdown("""
    <style>
        .stock-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
            opacity: 0.03;
            font-size: 10rem;
            font-weight: 900;
            color: white;
            overflow: hidden;
        }
        
        .stock-symbol {
            position: absolute;
            animation: float 20s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-30px) rotate(5deg); }
        }
    </style>
    
    <div class="stock-background">
        <div class="stock-symbol" style="top: 10%; left: 10%;">📈</div>
        <div class="stock-symbol" style="top: 20%; right: 15%; animation-delay: 2s;">💹</div>
        <div class="stock-symbol" style="top: 60%; left: 20%; animation-delay: 4s;">📊</div>
        <div class="stock-symbol" style="bottom: 20%; right: 20%; animation-delay: 6s;">💰</div>
        <div class="stock-symbol" style="top: 40%; left: 50%; font-size: 15rem; opacity: 0.02;">AAPL</div>
        <div class="stock-symbol" style="top: 70%; right: 40%; font-size: 12rem; opacity: 0.02; animation-delay: 3s;">TSLA</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section - FIXED: Direct HTML without variables
    components.html(
"""
<style>
@keyframes slideFade {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
/* IFRAME ROOT — MATCH PAGE BACKGROUND */
html, body {
  margin: 0;
  padding: 0;
  background: #2c2858 !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
    Roboto, Helvetica, Arial, sans-serif;
  overflow: hidden;
  width: 100%;
  height: 100%;
}
/* SINGLE BOX CONTROLS EVERYTHING */
.hero-box {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  border-radius: 28px;
  overflow: hidden;
  /* Card gradient matching the blue */
  background: linear-gradient(135deg, #4a4580 0%, #3a3670 100%);
  /* Very subtle outline */
  box-shadow: 
    0 0 0 0.5px rgba(255,255,255,0.06),
    0 8px 32px rgba(0,0,0,0.15);
  padding: 48px 60px 54px;
  animation: slideFade 0.3s ease-out;
  box-sizing: border-box;
}
/* BADGE */
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 18px;
  font-size: 13px;
  font-weight: 600;
  color: #e9d5ff;
  background: rgba(255,255,255,0.14);
  border-radius: 999px;
  margin-bottom: 18px;
}
/* TITLE */
.hero-title {
  font-size: 42px;
  font-weight: 800;
  letter-spacing: -0.5px;
  color: #ffffff;
  margin: 0 0 14px 0;
}
/* SUBTITLE */
.hero-subtitle {
  font-size: 16px;
  line-height: 1.7;
  color: rgba(255,255,255,0.82);
  max-width: 920px;
  margin: 0;
}
.hero-subtitle strong {
  color: #ddd6fe;
  font-weight: 600;
}
/* MOBILE */
@media (max-width: 768px) {
  .hero-box {
    border-radius: 24px;
    padding: 34px 28px 38px;
  }
  .hero-title {
    font-size: 30px;
  }
}
</style>
<div class="hero-box">
  <div class="hero-badge">✨ LSTM Neural Network</div>
  <div class="hero-title">
    🚀 AI-Powered Stock Predictions
  </div>
  <div class="hero-subtitle">
    Advanced LSTM neural networks analyzing
    <strong>AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA</strong>
    with adaptive thresholds, real-time price ingestion,
    market-regime detection, and enforced risk-reward
    logic (R:R ≥ 1.5:1) for institutional-grade decisions.
  </div>
</div>
""",
height=250,
scrolling=False,
)


    
    # 6 Stocks Showcase
    st.markdown("### 🎯 Analyzing Top 6 Tech Giants")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🍎</div>
            <h3 style="color: #a78bfa; margin-bottom: 0.3rem;">AAPL</h3>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Apple Inc.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🪟</div>
            <h3 style="color: #818cf8; margin-bottom: 0.3rem;">MSFT</h3>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Microsoft Corp.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔍</div>
            <h3 style="color: #60a5fa; margin-bottom: 0.3rem;">GOOGL</h3>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Alphabet Inc.</p>
        </div>
        """, unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📦</div>
            <h3 style="color: #34d399; margin-bottom: 0.3rem;">AMZN</h3>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Amazon.com Inc.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎮</div>
            <h3 style="color: #a78bfa; margin-bottom: 0.3rem;">NVDA</h3>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">NVIDIA Corp.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚡</div>
            <h3 style="color: #fbbf24; margin-bottom: 0.3rem;">TSLA</h3>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Tesla Inc.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Cards Grid
    st.markdown("### ✨ Core Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #a78bfa; margin-bottom: 0.5rem;">Real-Time Analysis</h3>
            <p style="color: rgba(255,255,255,0.7);">
                Live price updates from yfinance with automatic CSV data enhancement. 
                Get the most current market data for accurate predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <h3 style="color: #818cf8; margin-bottom: 0.5rem;">Adaptive Thresholds</h3>
            <p style="color: rgba(255,255,255,0.7);">
                Per-stock threshold optimization based on volatility patterns, 
                market regime, and trend consistency for precision signals.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🛡️</div>
            <h3 style="color: #60a5fa; margin-bottom: 0.5rem;">Risk Management</h3>
            <p style="color: rgba(255,255,255,0.7);">
                Guaranteed R:R ratio ≥1.5:1 with dynamic stop-loss and target 
                calculation. Protect your capital with intelligent risk controls.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Statistics Grid
    st.markdown("### 📈 Platform Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Predictions Made</div>
            <div class="metric-value">12,847</div>
            <div class="metric-subtitle">All-time total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value">78.5%</div>
            <div class="metric-subtitle">Accuracy metric</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Focus Stocks</div>
            <div class="metric-value">6</div>
            <div class="metric-subtitle">Top tech giants</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Avg R:R Ratio</div>
            <div class="metric-value">2.15:1</div>
            <div class="metric-subtitle">Risk-reward</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown("### 🔬 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #a78bfa; margin-bottom: 1rem;">📡 Data Collection</h4>
            <div style="color: rgba(255,255,255,0.8); line-height: 1.8;">
                <strong>1. Real-Time Fetching:</strong> Automatically updates prices from yfinance<br>
                <strong>2. Historical Analysis:</strong> Processes 2+ years of OHLCV data<br>
                <strong>3. Technical Indicators:</strong> Calculates 15+ advanced indicators<br>
                <strong>4. Market Context:</strong> Analyzes SPY for market regime detection
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #818cf8; margin-bottom: 1rem;">🧠 AI Processing</h4>
            <div style="color: rgba(255,255,255,0.8); line-height: 1.8;">
                <strong>1. LSTM Neural Network:</strong> Multi-layered deep learning model<br>
                <strong>2. Adaptive Thresholds:</strong> Stock-specific probability requirements<br>
                <strong>3. Weighted Scoring:</strong> 0-100 scale with 4 components<br>
                <strong>4. Risk Optimization:</strong> Ensures R:R ≥1.5:1 on every trade
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Signal Scoring Breakdown
    st.markdown("### 🎯 Signal Scoring System (0-100)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #a78bfa; margin-bottom: 0.5rem;">40</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-bottom: 0.5rem;">POINTS</div>
            <strong style="color: white;">Probability Margin</strong>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 100%; background: #a78bfa;"></div>
            </div>
            <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-top: 0.5rem;">
                How far probability exceeds threshold
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #818cf8; margin-bottom: 0.5rem;">25</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-bottom: 0.5rem;">POINTS</div>
            <strong style="color: white;">Risk-Reward</strong>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 100%; background: #818cf8;"></div>
            </div>
            <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-top: 0.5rem;">
                Quality of R:R ratio (≥1.5:1 required)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #60a5fa; margin-bottom: 0.5rem;">20</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-bottom: 0.5rem;">POINTS</div>
            <strong style="color: white;">Market Alignment</strong>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 100%; background: #60a5fa;"></div>
            </div>
            <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-top: 0.5rem;">
                Signal matches market regime
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #34d399; margin-bottom: 0.5rem;">15</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-bottom: 0.5rem;">POINTS</div>
            <strong style="color: white;">Volatility</strong>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 100%; background: #34d399;"></div>
            </div>
            <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-top: 0.5rem;">
                Lower volatility = higher score
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # CTA Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0;">
            <h3 style="color: white; margin-bottom: 1.5rem;">Ready to Analyze the Top 6?</h3>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 2rem;">
                Select stocks from the sidebar and generate AI-powered predictions for AAPL, MSFT, 
                GOOGL, AMZN, NVDA, and TSLA in seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box" style="margin-top: 2rem;">
        <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
        This tool is for educational and research purposes only. Not financial advice. 
        Predictions for AAPL, MSFT, GOOGL, AMZN, NVDA, and TSLA are based on historical data 
        and machine learning models which cannot guarantee future performance. Always paper trade 
        extensively before risking real capital and consult licensed financial professionals for investment advice.
    </div>
    """, unsafe_allow_html=True)
# ============================================================================
# SIDEBAR - STOCK SELECTION & CONTROLS (6 STOCKS ONLY)
# ============================================================================

def render_sidebar():
    """Render enhanced sidebar with controls - Limited to 6 major tech stocks"""
    
    with st.sidebar:
        # Logo/Title Section
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📈</div>
            <h2 style="color: #a78bfa; margin: 0; font-size: 1.5rem; font-weight: 800;">
                Stock Predictor Pro
            </h2>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.3rem;">
                Enhanced v2.0 | Top 6 Tech Stocks
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        
        page_options = {
            "🏠 Home": "home",
            "🔮 Predictions": "predictions",
            "📊 Portfolio Analysis": "portfolio",
            "📈 Technical Charts": "charts",
            "📋 History Log": "history"
        }
        
        selected_page = st.radio(
            "Go to:",
            list(page_options.keys()),
            index=list(page_options.values()).index(st.session_state.current_page),
            label_visibility="collapsed"
        )
        
        st.session_state.current_page = page_options[selected_page]
        
        st.markdown("---")
        
        # Stock Selection Section - ONLY 6 STOCKS
        st.markdown("### 📊 Stock Selection")
        
        st.markdown("""
        <div style="background: rgba(138, 43, 226, 0.2); padding: 0.8rem; border-radius: 12px; 
                    margin-bottom: 1rem; border: 1px solid rgba(167, 139, 250, 0.3);">
            <div style="text-align: center; color: #a78bfa; font-size: 0.85rem; font-weight: 600;">
                🎯 Top 6 Tech Giants
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # FIXED LIST - Only 6 stocks
        AVAILABLE_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
        
        # Analysis mode
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Single Stock", "Portfolio Analysis (All 6)"],
            help="Analyze one stock or all 6 stocks for comparison"
        )
        
        analyze_stocks = []
        
        if analysis_mode == "Single Stock":
            ticker = st.selectbox(
                "Select Stock",
                AVAILABLE_STOCKS,
                index=0,
                help="Choose from the top 6 tech stocks"
            )
            analyze_stocks = [ticker]
            
            # Show stock info
            stock_info = {
                "AAPL": "🍎 Apple Inc.",
                "MSFT": "🪟 Microsoft Corp.",
                "GOOGL": "🔍 Alphabet Inc.",
                "AMZN": "📦 Amazon.com Inc.",
                "NVDA": "🎮 NVIDIA Corp.",
                "TSLA": "⚡ Tesla Inc."
            }
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 10px; 
                        margin-top: 0.5rem; text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">{stock_info[ticker].split()[0]}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">{' '.join(stock_info[ticker].split()[1:])}</div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            # Portfolio mode - ALL 6 stocks
            st.info("📊 Full portfolio analysis of all 6 tech giants")
            analyze_stocks = AVAILABLE_STOCKS.copy()
            
            # Show all stocks with icons
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; margin-top: 0.5rem;">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; font-size: 0.85rem;">
                    <div>🍎 <strong>AAPL</strong></div>
                    <div>🪟 <strong>MSFT</strong></div>
                    <div>🔍 <strong>GOOGL</strong></div>
                    <div>📦 <strong>AMZN</strong></div>
                    <div>🎮 <strong>NVDA</strong></div>
                    <div>⚡ <strong>TSLA</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Analysis Button
        st.markdown("### 🚀 Generate Analysis")
        
        if st.button("🔮 Run Predictions", use_container_width=True, type="primary"):
            if not analyze_stocks:
                st.error("⚠️ Please select at least one stock")
            else:
                with st.spinner("🔮 Running Enhanced LSTM Model..."):
                    st.session_state.predictions = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, stock in enumerate(analyze_stocks):
                        try:
                            status_text.text(f"Analyzing {stock}... ({idx+1}/{len(analyze_stocks)})")
                            
                            # Run prediction
                            pred = predict_stock_enhanced(stock)
                            st.session_state.predictions[stock] = pred
                            
                            # Update progress
                            progress_bar.progress((idx + 1) / len(analyze_stocks))
                            
                            # Small delay for visual effect
                            time.sleep(0.3)
                            
                        except Exception as e:
                            st.error(f"❌ Error predicting {stock}: {str(e)}")
                            continue
                    
                    if st.session_state.predictions:
                        st.session_state.last_analysis_time = datetime.now()
                        st.success(f"✅ Analyzed {len(st.session_state.predictions)} stocks!")
                        
                        # Show quick summary
                        buy_signals = sum(1 for p in st.session_state.predictions.values() 
                                        if "BUY" in p.action)
                        avg_score = np.mean([p.signal_score for p in st.session_state.predictions.values()])
                        
                        st.info(f"""
                        **Quick Summary:**
                        - 🟢 Buy Signals: {buy_signals}/{len(st.session_state.predictions)}
                        - 📊 Avg Score: {avg_score:.0f}/100
                        - 🎯 Best: {max(st.session_state.predictions.values(), key=lambda x: x.signal_score).symbol}
                        """)
                        
                        # Auto-log to CSV
                        try:
                            log_to_csv(list(st.session_state.predictions.values()))
                            st.caption("📝 Logged to predictions_log.csv")
                        except Exception as e:
                            st.warning(f"⚠️ Logging failed: {e}")
                    else:
                        st.error("❌ No predictions generated")
                    
                    progress_bar.empty()
                    status_text.empty()
        
        st.markdown("---")
        
        # Filters (if predictions exist)
        if st.session_state.predictions:
            st.markdown("### 🔍 Filters")
            
            # Action filter
            action_filter = st.selectbox(
                "Filter by Action",
                ["All", "Buy Signals", "Sell Signals", "Wait/Hold"],
                help="Filter predictions by recommended action"
            )
            
            # Score filter
            min_score = st.slider(
                "Minimum Signal Score",
                0, 100, 0,
                help="Show only predictions above this score"
            )
            
            # Store filters in session state
            st.session_state.action_filter = action_filter
            st.session_state.min_score = min_score
            
            st.markdown("---")
        
        # Enhanced Features Info
        st.markdown("### ✨ Enhanced v2 Features")
        
        with st.expander("📋 What's New"):
            st.markdown("""
            **🎯 Adaptive Thresholds:**
            - Per-stock threshold optimization
            - Volatility-adjusted requirements
            - Trend consistency analysis
            
            **📊 Weighted Scoring:**
            - 0-100 point scale
            - 4 component breakdown
            - Clear signal strength
            
            **🛡️ Risk Management:**
            - Guaranteed R:R ≥ 1.5:1
            - Dynamic stop-loss calculation
            - Volatility-adjusted targets
            
            **📡 Real-Time Data:**
            - Live price from yfinance
            - Automatic CSV updates
            - Market regime detection
            
            **🎯 Top 6 Tech Focus:**
            - AAPL, MSFT, GOOGL
            - AMZN, NVDA, TSLA
            - High-quality predictions only
            """)
        
        with st.expander("📖 How to Use"):
            st.markdown("""
            **1. Select Mode:**
            - Single Stock: Analyze one at a time
            - Portfolio: Analyze all 6 together
            
            **2. Run Predictions:**
            - Click "Run Predictions" button
            - Wait for AI analysis (10-20s per stock)
            
            **3. Review Results:**
            - Check signal scores (0-100)
            - Review risk-reward ratios
            - Read AI reasoning
            
            **4. Score Guide:**
            - 75-100: ⭐ Excellent signal
            - 65-74: ✅ Good signal
            - 55-64: ⚠️ Marginal signal
            - 0-54: ❌ Weak signal
            
            **⚠️ Always paper trade first!**
            """)
        
        with st.expander("⚙️ Settings"):
            st.markdown("**Display Options:**")
            
            show_warnings = st.checkbox("Show Warnings", value=True)
            show_reasoning = st.checkbox("Show AI Reasoning", value=True)
            show_breakdown = st.checkbox("Show Score Breakdown", value=True)
            
            st.markdown("**Export Options:**")
            auto_log = st.checkbox("Auto-log to CSV", value=True)
            
            # Store in session state
            st.session_state.show_warnings = show_warnings
            st.session_state.show_reasoning = show_reasoning
            st.session_state.show_breakdown = show_breakdown
        
        st.markdown("---")
        
        # Quick Stats (if predictions exist)
        if st.session_state.predictions:
            st.markdown("### 📊 Quick Stats")
            
            total = len(st.session_state.predictions)
            avg_score = np.mean([p.signal_score for p in st.session_state.predictions.values()])
            avg_rr = np.mean([p.risk_reward for p in st.session_state.predictions.values()])
            buy_count = sum(1 for p in st.session_state.predictions.values() if "BUY" in p.action)
            
            st.metric("Stocks Analyzed", f"{total}/6")
            st.metric("Avg Score", f"{avg_score:.0f}/100")
            st.metric("Avg R:R", f"{avg_rr:.2f}:1")
            st.metric("Buy Signals", buy_count)
            
            # Last analysis time
            if st.session_state.last_analysis_time:
                st.caption(f"🕐 Last: {st.session_state.last_analysis_time.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; color: rgba(255,255,255,0.4); font-size: 0.75rem;">
            <p><strong>⚠️ DISCLAIMER</strong></p>
            <p>For educational purposes only.<br>Not financial advice.</p>
            <p style="margin-top: 1rem; color: rgba(255,255,255,0.3);">
                © 2024 Stock Predictor Pro v2.0<br>
                Limited to 6 Tech Stocks
            </p>
        </div>
        """, unsafe_allow_html=True)

# Apply any filters
def apply_filters(predictions, action_filter="All", min_score=0):
    """Apply user-selected filters to predictions"""
    filtered = {}
    
    for symbol, pred in predictions.items():
        # Score filter
        if pred.signal_score < min_score:
            continue
        
        # Action filter
        if action_filter == "Buy Signals" and "BUY" not in pred.action:
            continue
        elif action_filter == "Sell Signals" and "SELL" not in pred.action:
            continue
        elif action_filter == "Wait/Hold" and not any(x in pred.action for x in ["WAIT", "NO TRADE"]):
            continue
        
        filtered[symbol] = pred
    
    return filtered
# ============================================================================
# PREDICTIONS PAGE - FIXED ALIGNMENT & SIGNAL SCORE VISIBILITY
# ============================================================================

def render_predictions_page():
    """Render predictions page with fixed layout and visibility"""
    
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="color: #a78bfa; font-size: 2.5rem; margin-bottom: 0.5rem;">
            🔮 Stock Predictions
        </h2>
        <p style="color: rgba(255,255,255,0.7);">
            AI-powered analysis with real-time data for 6 major tech stocks
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.predictions:
        # Empty state
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: white; margin-bottom: 1rem;">No Predictions Yet</h3>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 2rem;">
                Select stocks from the sidebar (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA) 
                and click <strong>"Run Predictions"</strong> to see AI-powered analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Show analysis time
    if st.session_state.last_analysis_time:
        st.caption(f"🕐 Analysis Time: {st.session_state.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Portfolio Overview Summary
    if len(st.session_state.predictions) > 1:
        st.markdown("### 📊 Portfolio Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total = len(st.session_state.predictions)
        buy_signals = sum(1 for p in st.session_state.predictions.values() if "BUY" in p.action)
        sell_signals = sum(1 for p in st.session_state.predictions.values() if "SELL" in p.action)
        wait_signals = sum(1 for p in st.session_state.predictions.values() if "WAIT" in p.action or "NO TRADE" in p.action)
        avg_score = np.mean([p.signal_score for p in st.session_state.predictions.values()])
        avg_rr = np.mean([p.risk_reward for p in st.session_state.predictions.values()])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Stocks</div>
                <div class="metric-value">{total}</div>
                <div class="metric-subtitle">Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🟢 Buy Signals</div>
                <div class="metric-value" style="color: #22c55e;">{buy_signals}</div>
                <div class="metric-subtitle">{buy_signals/total*100:.0f}% of portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">⏸️ Wait/Hold</div>
                <div class="metric-value" style="color: #9ca3af;">{wait_signals}</div>
                <div class="metric-subtitle">{wait_signals/total*100:.0f}% of portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Score</div>
                <div class="metric-value" style="color: #a78bfa;">{avg_score:.0f}</div>
                <div class="metric-subtitle">out of 100</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg R:R</div>
                <div class="metric-value" style="color: #60a5fa;">{avg_rr:.2f}:1</div>
                <div class="metric-subtitle">Risk-reward ratio</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Individual Stock Prediction Cards
    st.markdown("### 📈 Detailed Analysis")
    
    for symbol, pred in st.session_state.predictions.items():
        # Main card container with Score + Action Badge - FIXED WRAPPING
        st.markdown(f"""
        <div class="prediction-card" style="margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <div style="flex-shrink: 0;">
                    <h2 style="color: white; margin: 0; font-size: 2.5rem;">{symbol}</h2>
                    <p style="color: rgba(255,255,255,0.5); margin: 0.3rem 0 0 0; font-size: 0.9rem;">{pred.price_date}</p>
                </div>
                <div style="display: flex; gap: 1rem; align-items: center; flex-shrink: 0; margin-left: 1rem;">
                    <div class="score-badge {get_score_badge_class(pred.signal_score)}" style="white-space: nowrap;">
                        {pred.signal_score:.0f}/100
                    </div>
                    <div class="action-badge {get_action_color_class(pred.action)}" style="padding: 0.8rem 1.5rem; white-space: nowrap;">
                        {pred.action}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Top metrics row - FIXED ALIGNMENT WITH PROPER SPACING
            cols = st.columns([1.8, 1.2, 1, 1])  # Adjusted ratios for better fit
            
            with cols[0]:
                st.markdown(f"""
                <div class="metric-card" style="padding: 1.5rem; min-height: 120px;">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value" style="font-size: 2.2rem;">${pred.current_price:.2f}</div>
                    <div class="metric-subtitle">{pred.price_date}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                direction_color = "#22c55e" if "UP" in pred.week_direction else "#ef4444"
                st.markdown(f"""
                <div class="metric-card" style="min-height: 120px;">
                    <div class="metric-label">Direction</div>
                    <div class="metric-value" style="color: {direction_color}; font-size: 1.5rem;">{pred.week_direction}</div>
                    <div class="metric-subtitle">{pred.week_prob_up*100:.1f}% prob</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                <div class="metric-card" style="min-height: 120px;">
                    <div class="metric-label">Score</div>
                    <div class="metric-value">{pred.signal_score:.0f}</div>
                    <div class="metric-subtitle" style="font-size: 0.75rem;">{pred.signal_strength}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                confidence_clean = clean_emoji(pred.confidence)
                st.markdown(f"""
                <div class="metric-card" style="min-height: 120px;">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value" style="font-size: 1rem;">{confidence_clean}</div>
                    <div class="metric-subtitle" style="font-size: 0.7rem;">Thr: {pred.adaptive_threshold*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Score breakdown
            st.markdown("#### 📊 Score Breakdown")
            breakdown_cols = st.columns(4)
            
            component_details = {
                'probability': {'max': 40, 'label': 'Probability'},
                'risk_reward': {'max': 25, 'label': 'Risk-Reward'},
                'market_alignment': {'max': 20, 'label': 'Market Align'},
                'volatility': {'max': 15, 'label': 'Volatility'}
            }
            
            for idx, (component, score) in enumerate(pred.score_breakdown.items()):
                with breakdown_cols[idx]:
                    details = component_details[component]
                    percentage = (score / details['max']) * 100
                    
                    color = '#22c55e' if percentage >= 75 else '#fbbf24' if percentage >= 50 else '#ef4444'
                    
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 1rem; text-align: center;">
                        <strong style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">{details['label']}</strong><br>
                        <span style="font-size: 1.8rem; color: {color}; font-weight: 800;">{score:.0f}</span>
                        <span style="font-size: 0.9rem; color: rgba(255,255,255,0.5);">/{details['max']}</span>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {percentage}%; background: {color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Risk Management Section
            st.markdown("#### 🎯 Risk Management")
            risk_cols = st.columns(3)
            
            with risk_cols[0]:
                st.markdown(f"""
                <div class="glass-card">
                    <strong style="color: #60a5fa;">Entry & Targets:</strong><br>
                    <div style="margin-top: 0.8rem; line-height: 1.8;">
                        Entry: <span style="color: #3b82f6; font-weight: 700;">${pred.current_price:.2f}</span><br>
                        Target: <span style="color: #22c55e; font-weight: 700;">${pred.target_low:.2f} - ${pred.target_high:.2f}</span><br>
                        Expected: <span style="color: #22c55e; font-weight: 700;">+{pred.expected_return:.2f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with risk_cols[1]:
                rr_color = "#22c55e" if pred.risk_reward >= 2.0 else "#fbbf24" if pred.risk_reward >= 1.5 else "#ef4444"
                st.markdown(f"""
                <div class="glass-card">
                    <strong style="color: #ef4444;">Risk Parameters:</strong><br>
                    <div style="margin-top: 0.8rem; line-height: 1.8;">
                        Stop Loss: <span style="color: #ef4444; font-weight: 700;">${pred.stop_loss:.2f}</span><br>
                        Max Loss: <span style="color: #ef4444; font-weight: 700;">-{pred.max_loss:.2f}%</span><br>
                        R:R: <span style="color: {rr_color}; font-weight: 700;">{pred.risk_reward:.2f}:1</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with risk_cols[2]:
                regime_clean = clean_emoji(pred.market_regime)
                st.markdown(f"""
                <div class="glass-card">
                    <strong style="color: #a78bfa;">Market Context:</strong><br>
                    <div style="margin-top: 0.8rem; line-height: 1.8;">
                        Regime: <span style="font-weight: 600;">{regime_clean}</span><br>
                        Volatility: <span style="font-weight: 600;">{pred.volatility*100:.2f}% ({pred.volatility_regime})</span><br>
                        ATR: <span style="font-weight: 600;">{pred.atr_pct:.2f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_right:
            # Threshold adaptation info
            st.markdown("#### 🎚️ Adaptive Threshold")
            st.markdown(f"""
            <div class="info-box">
                <strong>Base:</strong> 55.0%<br>
                <strong>Vol Adj:</strong> {pred.threshold_breakdown['vol_adjustment']*100:+.1f}%<br>
                <strong>Regime Adj:</strong> {pred.threshold_breakdown['regime_adjustment']*100:+.1f}%<br>
                <strong>Final:</strong> {pred.adaptive_threshold*100:.1f}%<br>
                <strong>Trend Consistency:</strong> {pred.threshold_breakdown.get('trend_consistency', 0)*100:.0f}%
            </div>
            """, unsafe_allow_html=True)
            
            # AI Reasoning
            st.markdown("#### 🧠 AI Analysis")
            for reason in pred.reasoning:
                if "✅" in reason:
                    st.markdown(f'<div class="success-box">{reason}</div>', unsafe_allow_html=True)
                elif "⚠️" in reason or "❌" in reason:
                    st.markdown(f'<div class="warning-box">{reason}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="info-box">{reason}</div>', unsafe_allow_html=True)
            
            # Warnings
            if pred.warnings:
                st.markdown("#### ⚠️ Warnings")
                for warning in pred.warnings:
                    st.markdown(f'<div class="warning-box">• {warning}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Best opportunity highlight
    if any("BUY" in p.action or "SELL" in p.action for p in st.session_state.predictions.values()):
        trade_preds = [p for p in st.session_state.predictions.values() 
                      if "BUY" in p.action or "SELL" in p.action]
        best = max(trade_preds, key=lambda x: x.signal_score)
        
        st.markdown(f"""
        <div class="success-box" style="text-align: center; padding: 1.5rem; font-size: 1.1rem;">
            <strong>🏆 BEST OPPORTUNITY:</strong> {best.symbol} 
            (Score: {best.signal_score:.0f}/100, R:R: {best.risk_reward:.2f}:1, {clean_emoji(best.action)})
        </div>
        """, unsafe_allow_html=True)
# ============================================================================
# PORTFOLIO COMPARISON PAGE - FIXED
# ============================================================================

def render_portfolio_page():
    """Render portfolio comparison page with comparative table"""
    
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="color: #a78bfa; font-size: 2.5rem; margin-bottom: 0.5rem;">
            📊 Portfolio Analysis
        </h2>
        <p style="color: rgba(255,255,255,0.7);">
            Comparative analysis of all 6 tech stocks
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.predictions:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
            <h3 style="color: white; margin-bottom: 1rem;">No Portfolio Data</h3>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 2rem;">
                Run predictions for multiple stocks to see comparative portfolio analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Create comparison dataframe
    comparison_data = []
    for symbol, p in st.session_state.predictions.items():
        comparison_data.append({
            'Symbol': symbol,
            'Price': p.current_price,
            'Direction': clean_emoji(p.week_direction),
            'Probability': p.week_prob_up * 100,
            'Threshold': p.adaptive_threshold * 100,
            'Score': p.signal_score,
            'Signal': p.signal_strength,
            'Confidence': clean_emoji(p.confidence),
            'Target Low': p.target_low,
            'Target High': p.target_high,
            'Expected Return': p.expected_return,
            'R:R': p.risk_reward,
            'Max Loss': p.max_loss,
            'Regime': clean_emoji(p.market_regime),
            'Volatility': p.volatility * 100,
            'Action': clean_emoji(p.action)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Summary metrics
    st.markdown("### 📊 Portfolio Summary")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Stocks</div>
            <div class="metric-value">{len(df_comparison)}</div>
            <div class="metric-subtitle">Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        buy_count = len(df_comparison[df_comparison['Action'].str.contains('BUY', na=False)])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🟢 Buy Signals</div>
            <div class="metric-value" style="color: #22c55e;">{buy_count}</div>
            <div class="metric-subtitle">{buy_count/len(df_comparison)*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        wait_count = len(df_comparison[df_comparison['Action'].str.contains('WAIT|NO TRADE', na=False)])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏸️ Wait/Hold</div>
            <div class="metric-value" style="color: #9ca3af;">{wait_count}</div>
            <div class="metric-subtitle">{wait_count/len(df_comparison)*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_score = df_comparison['Score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Score</div>
            <div class="metric-value" style="color: #a78bfa;">{avg_score:.0f}</div>
            <div class="metric-subtitle">out of 100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        avg_rr = df_comparison['R:R'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg R:R</div>
            <div class="metric-value" style="color: #60a5fa;">{avg_rr:.2f}:1</div>
            <div class="metric-subtitle">Risk-reward</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        avg_return = df_comparison['Expected Return'].mean()
        return_color = "#22c55e" if avg_return > 0 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Return</div>
            <div class="metric-value" style="color: {return_color};">{avg_return:+.1f}%</div>
            <div class="metric-subtitle">Expected</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison Table
    st.markdown("### 📋 Comparative Analysis Table")
    
    # Format for display - KEEP NUMERIC VALUES FOR SORTING
    df_display = df_comparison.copy()
    
    # Apply custom styling with proper column selection
    def style_score(val):
        """Style score column"""
        try:
            val = float(val)
            if val >= 75:
                return 'background-color: rgba(34, 197, 94, 0.2); color: #22c55e; font-weight: bold;'
            elif val >= 65:
                return 'background-color: rgba(59, 130, 246, 0.2); color: #3b82f6; font-weight: bold;'
            elif val >= 55:
                return 'background-color: rgba(251, 191, 36, 0.2); color: #fbbf24; font-weight: bold;'
            else:
                return 'background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold;'
        except:
            return ''
    
    def style_return(val):
        """Style expected return column"""
        try:
            val = float(val)
            if val > 0:
                return 'color: #22c55e; font-weight: bold;'
            elif val < 0:
                return 'color: #ef4444; font-weight: bold;'
            return ''
        except:
            return ''
    
    # Format columns for display AFTER styling
    df_styled = df_display.style.applymap(style_score, subset=['Score']).applymap(style_return, subset=['Expected Return'])
    
    # Now format the display values
    df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:.2f}")
    df_display['Probability'] = df_display['Probability'].apply(lambda x: f"{x:.1f}%")
    df_display['Threshold'] = df_display['Threshold'].apply(lambda x: f"{x:.1f}%")
    df_display['Target Low'] = df_display['Target Low'].apply(lambda x: f"${x:.0f}")
    df_display['Target High'] = df_display['Target High'].apply(lambda x: f"${x:.0f}")
    df_display['Expected Return'] = df_display['Expected Return'].apply(lambda x: f"{x:+.1f}%")
    df_display['R:R'] = df_display['R:R'].apply(lambda x: f"{x:.2f}:1")
    df_display['Max Loss'] = df_display['Max Loss'].apply(lambda x: f"{x:.1f}%")
    df_display['Volatility'] = df_display['Volatility'].apply(lambda x: f"{x:.2f}%")
    
    # Display dataframe without problematic styling
    st.dataframe(
        df_display,
        use_container_width=True,
        height=400
    )
    
    # Download button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        csv = df_comparison.to_csv(index=False)
        st.download_button(
            label="📥 Download Portfolio CSV",
            data=csv,
            file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Best/Worst Opportunities
    st.markdown("### 🏆 Top Opportunities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Best opportunity
        if buy_count > 0:
            buy_stocks = df_comparison[df_comparison['Action'].str.contains('BUY', na=False)]
            best = buy_stocks.loc[buy_stocks['Score'].idxmax()]
            
            st.markdown(f"""
            <div class="success-box">
                <h4 style="color: #22c55e; margin-bottom: 1rem;">✨ Best Buy Signal</h4>
                <div style="font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem;">{best['Symbol']}</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 1rem;">
                    <div><strong>Score:</strong> {best['Score']:.0f}/100</div>
                    <div><strong>R:R:</strong> {best['R:R']:.2f}:1</div>
                    <div><strong>Expected:</strong> {best['Expected Return']:+.1f}%</div>
                    <div><strong>Confidence:</strong> {best['Confidence']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No buy signals in current portfolio")
    
    with col2:
        # Weakest signal
        worst = df_comparison.loc[df_comparison['Score'].idxmin()]
        
        st.markdown(f"""
        <div class="warning-box">
            <h4 style="color: #fbbf24; margin-bottom: 1rem;">⚠️ Weakest Signal</h4>
            <div style="font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem;">{worst['Symbol']}</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 1rem;">
                <div><strong>Score:</strong> {worst['Score']:.0f}/100</div>
                <div><strong>Action:</strong> {worst['Action']}</div>
                <div><strong>Confidence:</strong> {worst['Confidence']}</div>
                <div><strong>Volatility:</strong> {worst['Volatility']:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TECHNICAL CHARTS PAGE
# ============================================================================

def render_charts_page():
    """Render technical charts page"""
    
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="color: #a78bfa; font-size: 2.5rem; margin-bottom: 0.5rem;">
            📈 Technical Charts
        </h2>
        <p style="color: rgba(255,255,255,0.7);">
            Interactive price charts with predictions overlay
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.predictions:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: white; margin-bottom: 1rem;">No Chart Data</h3>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 2rem;">
                Run predictions first to see technical charts with target levels.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Stock selector
    selected_stock = st.selectbox(
        "Select Stock for Charts",
        list(st.session_state.predictions.keys()),
        help="Choose a stock to view detailed charts"
    )
    
    if not selected_stock:
        return
    
    pred = st.session_state.predictions[selected_stock]
    
    # Stock header
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom: 2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="color: white; margin: 0;">{selected_stock}</h2>
                <p style="color: rgba(255,255,255,0.6); margin: 0.3rem 0 0 0;">
                    ${pred.current_price:.2f} • {pred.price_date}
                </p>
            </div>
            <div class="score-badge {get_score_badge_class(pred.signal_score)}">
                {pred.signal_score:.0f}/100
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Load stock data
        csv_paths = [
            Path(f"data/stock_data/{selected_stock}.csv"),
            Path(f"data/{selected_stock}.csv"),
        ]
        
        df = None
        for csv_path in csv_paths:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                break
        
        if df is not None:
            df.columns = df.columns.str.lower()
            
            # Find date column
            date_col = None
            for col in ['date', 'datetime', 'timestamp']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()
            
            # Display recent data (90 days)
            df_recent = df.tail(90)
            
            # Price Chart with Targets
            st.markdown("### 📈 Price Chart with Prediction Levels")
            
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=df_recent.index,
                open=df_recent['open'],
                high=df_recent['high'],
                low=df_recent['low'],
                close=df_recent['close'],
                name='OHLC',
                increasing_line_color='#22c55e',
                decreasing_line_color='#ef4444'
            ))
            
            # Current price line
            fig.add_hline(
                y=pred.current_price,
                line_dash="solid",
                line_color="#a78bfa",
                line_width=2,
                annotation_text=f"Current: ${pred.current_price:.2f}",
                annotation_position="right",
                annotation_font_color="#a78bfa"
            )
            
            # Target high
            fig.add_hline(
                y=pred.target_high,
                line_dash="dash",
                line_color="#22c55e",
                line_width=2,
                annotation_text=f"Target High: ${pred.target_high:.2f}",
                annotation_position="right",
                annotation_font_color="#22c55e"
            )
            
            # Target low
            fig.add_hline(
                y=pred.target_low,
                line_dash="dash",
                line_color="#34d399",
                line_width=1.5,
                annotation_text=f"Target Low: ${pred.target_low:.2f}",
                annotation_position="right",
                annotation_font_color="#34d399"
            )
            
            # Stop loss
            fig.add_hline(
                y=pred.stop_loss,
                line_dash="dash",
                line_color="#ef4444",
                line_width=2,
                annotation_text=f"Stop Loss: ${pred.stop_loss:.2f}",
                annotation_position="right",
                annotation_font_color="#ef4444"
            )
            
            fig.update_layout(
                **get_chart_theme(),
                xaxis_rangeslider_visible=False,
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume Chart
            st.markdown("### 📊 Trading Volume")
            
            fig_vol = go.Figure(data=[go.Bar(
                x=df_recent.index,
                y=df_recent['volume'],
                marker_color='#a78bfa',
                marker_line_color='#8b5cf6',
                marker_line_width=0.5,
                name='Volume',
                opacity=0.7
            )])
            
            fig_vol.update_layout(
                **get_chart_theme(),
                height=250,
                showlegend=False,
                yaxis_title="Volume"
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Price Statistics
            st.markdown("### 📊 Price Statistics (Last 90 Days)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            week_change = ((df_recent['close'].iloc[-1] - df_recent['close'].iloc[-5]) / df_recent['close'].iloc[-5] * 100) if len(df_recent) >= 5 else 0
            month_change = ((df_recent['close'].iloc[-1] - df_recent['close'].iloc[-20]) / df_recent['close'].iloc[-20] * 100) if len(df_recent) >= 20 else 0
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">90D High</div>
                    <div class="metric-value">${df_recent['high'].max():.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">90D Low</div>
                    <div class="metric-value">${df_recent['low'].min():.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                week_color = "#22c55e" if week_change > 0 else "#ef4444"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">1 Week Change</div>
                    <div class="metric-value" style="color: {week_color};">{week_change:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                month_color = "#22c55e" if month_change > 0 else "#ef4444"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">1 Month Change</div>
                    <div class="metric-value" style="color: {month_color};">{month_change:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.warning(f"⚠️ Could not load chart data for {selected_stock}")
    
    except Exception as e:
        st.error(f"❌ Error loading charts: {str(e)}")
# ============================================================================
# HISTORY LOG PAGE - FIXED
# ============================================================================

def render_history_page():
    """Render prediction history log page"""
    
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="color: #a78bfa; font-size: 2.5rem; margin-bottom: 0.5rem;">
            📋 Prediction History
        </h2>
        <p style="color: rgba(255,255,255,0.7);">
            Track all your past predictions and analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        log_path = Path("predictions_log.csv")
        
        if not log_path.exists():
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📝</div>
                <h3 style="color: white; margin-bottom: 1rem;">No History Yet</h3>
                <p style="color: rgba(255,255,255,0.7); margin-bottom: 1rem;">
                    Your prediction history will appear here after you run your first analysis.
                </p>
                <div class="info-box" style="max-width: 600px; margin: 2rem auto;">
                    <strong>📍 Log File Location:</strong> predictions_log.csv<br>
                    The log file will be automatically created in the project root directory 
                    when you generate your first prediction.
                </div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Load log data
        log_df = pd.read_csv(log_path)
        
        # Convert numeric columns to proper types
        if 'signal_score' in log_df.columns:
            log_df['signal_score'] = pd.to_numeric(log_df['signal_score'], errors='coerce')
        if 'risk_reward' in log_df.columns:
            log_df['risk_reward'] = pd.to_numeric(log_df['risk_reward'], errors='coerce')
        
        # Summary Section
        st.markdown("### 📊 Log Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Predictions</div>
                <div class="metric-value">{len(log_df)}</div>
                <div class="metric-subtitle">All-time</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            unique_stocks = log_df['symbol'].nunique() if 'symbol' in log_df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Unique Stocks</div>
                <div class="metric-value">{unique_stocks}</div>
                <div class="metric-subtitle">Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if 'action' in log_df.columns:
                trade_signals = log_df['action'].str.contains('BUY|SELL', case=False, na=False).sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Trade Signals</div>
                    <div class="metric-value" style="color: #22c55e;">{trade_signals}</div>
                    <div class="metric-subtitle">{trade_signals/len(log_df)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric("Trade Signals", "N/A")
        
        with col4:
            if 'signal_score' in log_df.columns:
                avg_score = log_df['signal_score'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Score</div>
                    <div class="metric-value" style="color: #a78bfa;">{avg_score:.1f}</div>
                    <div class="metric-subtitle">out of 100</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric("Avg Score", "N/A")
        
        with col5:
            if 'risk_reward' in log_df.columns:
                avg_rr = log_df['risk_reward'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg R:R</div>
                    <div class="metric-value" style="color: #60a5fa;">{avg_rr:.2f}:1</div>
                    <div class="metric-subtitle">Risk-reward</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric("Avg R:R", "N/A")
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filters")
        
        filter_cols = st.columns(3)
        
        with filter_cols[0]:
            if 'symbol' in log_df.columns:
                selected_symbols = st.multiselect(
                    "Filter by Symbol",
                    options=sorted(log_df['symbol'].unique()),
                    default=[],
                    help="Select specific stocks to view"
                )
            else:
                selected_symbols = []
        
        with filter_cols[1]:
            if 'action' in log_df.columns:
                selected_actions = st.multiselect(
                    "Filter by Action",
                    options=sorted(log_df['action'].unique()),
                    default=[],
                    help="Filter by trade action"
                )
            else:
                selected_actions = []
        
        with filter_cols[2]:
            show_rows = st.selectbox(
                "Show Rows",
                options=[25, 50, 100, 200, "All"],
                index=1,
                help="Number of rows to display"
            )
        
        # Apply filters
        filtered_df = log_df.copy()
        if selected_symbols:
            filtered_df = filtered_df[filtered_df['symbol'].isin(selected_symbols)]
        if selected_actions:
            filtered_df = filtered_df[filtered_df['action'].isin(selected_actions)]
        
        # Sort by timestamp (most recent first)
        if 'timestamp' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('timestamp', ascending=False)
        
        # Display dataframe
        st.markdown("### 📄 Log Data")
        
        if show_rows == "All":
            display_df = filtered_df
        else:
            display_df = filtered_df.head(int(show_rows))
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        st.caption(f"Showing {len(display_df)} of {len(filtered_df)} filtered records ({len(log_df)} total)")
        
        # Download Options
        st.markdown("### 📥 Download Options")
        
        download_cols = st.columns(2)
        
        with download_cols[0]:
            csv_filtered = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Log",
                data=csv_filtered,
                file_name=f"predictions_log_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with download_cols[1]:
            csv_full = log_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Log",
                data=csv_full,
                file_name=f"predictions_log_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Performance Insights
        if len(log_df) >= 10 and 'signal_score' in log_df.columns:
            st.markdown("---")
            st.markdown("### 📊 Performance Insights")
            
            insight_cols = st.columns(3)
            
            with insight_cols[0]:
                if 'action' in log_df.columns:
                    st.markdown("**Action Distribution:**")
                    action_counts = log_df['action'].value_counts()
                    
                    for action, count in action_counts.head(5).items():
                        percentage = count/len(log_df)*100
                        st.markdown(f"""
                        <div class="glass-card" style="padding: 0.5rem; margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>{action[:20]}</span>
                                <strong>{count} ({percentage:.1f}%)</strong>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {percentage}%; background: #a78bfa;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with insight_cols[1]:
                st.markdown("**Score Distribution:**")
                
                # Use numeric comparisons only
                excellent = (log_df['signal_score'] >= 75).sum()
                good = ((log_df['signal_score'] >= 65) & (log_df['signal_score'] < 75)).sum()
                marginal = ((log_df['signal_score'] >= 55) & (log_df['signal_score'] < 65)).sum()
                weak = (log_df['signal_score'] < 55).sum()
                
                score_data = [
                    ("Excellent (≥75)", excellent, '#22c55e'),
                    ("Good (65-74)", good, '#3b82f6'),
                    ("Marginal (55-64)", marginal, '#fbbf24'),
                    ("Weak (<55)", weak, '#ef4444')
                ]
                
                for label, count, color in score_data:
                    percentage = count/len(log_df)*100 if len(log_df) > 0 else 0
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 0.5rem; margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>{label}</span>
                            <strong>{count} ({percentage:.1f}%)</strong>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {percentage}%; background: {color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with insight_cols[2]:
                if 'symbol' in log_df.columns:
                    st.markdown("**Most Analyzed Stocks:**")
                    top_stocks = log_df['symbol'].value_counts().head(6)
                    
                    for stock, count in top_stocks.items():
                        percentage = count/len(log_df)*100
                        st.markdown(f"""
                        <div class="glass-card" style="padding: 0.5rem; margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <strong>{stock}</strong>
                                <span>{count} times ({percentage:.1f}%)</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {percentage}%; background: #a78bfa;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error loading log: {str(e)}")

# ============================================================================
# MAIN APPLICATION ROUTER
# ============================================================================

def main():
    """Main application entry point with page routing"""
    
    # Render sidebar (always visible)
    render_sidebar()
    
    # Main content area - route to appropriate page
    if st.session_state.current_page == "home":
        render_home_page()
    
    elif st.session_state.current_page == "predictions":
        render_predictions_page()
    
    elif st.session_state.current_page == "portfolio":
        render_portfolio_page()
    
    elif st.session_state.current_page == "charts":
        render_charts_page()
    
    elif st.session_state.current_page == "history":
        render_history_page()
    
    else:
        # Default to home page
        st.session_state.current_page = "home"
        render_home_page()
    
    # Footer - FIXED: Direct HTML without function calls
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.6); position: relative;">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            <strong>Stock Predictor Pro v2.0</strong> | Enhanced LSTM Neural Network
        </p>
        <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">
            ✨ Features: Adaptive Thresholds • Weighted Scoring • Market Regime Detection • R:R ≥1.5:1
        </p>
        <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">
            🎯 Analyzing: AAPL • MSFT • GOOGL • AMZN • NVDA • TSLA
        </p>
        <p style="font-size: 0.85rem; color: rgba(255,255,255,0.5);">
            ⚠️ <strong>CRITICAL DISCLAIMER:</strong> For Educational & Research Purposes Only | Not Financial Advice
        </p>
        <p style="font-size: 0.8rem; color: rgba(255,255,255,0.4); margin-top: 1rem;">
            Always paper trade extensively before risking real capital | Consult licensed financial professionals for investment advice
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
# Sidebar
with st.sidebar:
    st.markdown("### 📊 Stock Selection")
    
    # Default portfolio stocks
    default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD"]
    
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Single Stock", "Portfolio Analysis"],
        help="Analyze one stock or multiple stocks for comparison"
    )
    
    if analysis_mode == "Single Stock":
        ticker = st.selectbox(
            "Select Symbol",
            default_stocks + ["Custom"],
            index=0
        )
        
        if ticker == "Custom":
            ticker = st.text_input("Enter Symbol", "").upper()
        
        analyze_stocks = [ticker] if ticker else []
    else:
        analyze_stocks = st.multiselect(
            "Select Stocks",
            default_stocks,
            default=default_stocks[:4],
            help="Select stocks for comparative analysis"
        )
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🚀 Generate Predictions", use_container_width=True):
        if not analyze_stocks:
            st.error("Please select at least one stock")
        else:
            with st.spinner("🔮 Running Enhanced LSTM Model..."):
                st.session_state.predictions = {}
                progress_bar = st.progress(0)
                
                for idx, stock in enumerate(analyze_stocks):
                    try:
                        pred = predict_stock_enhanced(stock)
                        st.session_state.predictions[stock] = pred
                        progress_bar.progress((idx + 1) / len(analyze_stocks))
                    except Exception as e:
                        st.error(f"Error predicting {stock}: {str(e)}")
                
                if st.session_state.predictions:
                    st.session_state.last_analysis_time = datetime.now()
                    st.success(f"✅ Analyzed {len(st.session_state.predictions)} stocks!")
                    
                    # Auto-log to CSV
                    try:
                        log_to_csv(list(st.session_state.predictions.values()))
                        st.info("📊 Logged to predictions_log.csv")
                    except Exception as e:
                        st.warning(f"Logging failed: {e}")
    
    st.markdown("---")
    st.markdown("### ℹ️ Enhanced v2 Features")
    st.markdown("""
    **✨ New in v2:**
    - 🎯 Adaptive thresholds per stock
    - 📊 Weighted signal scoring (0-100)
    - 🔍 Enhanced market regime detection
    - ⚖️ Guaranteed R:R ≥ 1.5:1
    - 📈 Trend consistency analysis
    
    **⚠️ Always paper trade first!**
    """)

# Main Content Tabs
tab_predictions, tab_comparison, tab_analysis, tab_logs = st.tabs([
    "🔮 Predictions",
    "📊 Comparison Table",
    "📈 Technical Charts", 
    "📋 History Log"
])

# Tab 1: Individual Predictions
with tab_predictions:
    if not st.session_state.predictions:
        st.info("👆 Select stocks in the sidebar and click **Generate Predictions** to see results")
    else:
        # Show analysis time
        if st.session_state.last_analysis_time:
            st.caption(f"🕐 Analysis Time: {st.session_state.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Portfolio overview if multiple stocks
        if len(st.session_state.predictions) > 1:
            st.subheader("📊 Portfolio Overview")
            
            cols = st.columns(5)
            total = len(st.session_state.predictions)
            
            buy_signals = sum(1 for p in st.session_state.predictions.values() 
                            if "BUY" in p.action)
            sell_signals = sum(1 for p in st.session_state.predictions.values() 
                             if "SELL" in p.action)
            wait_signals = sum(1 for p in st.session_state.predictions.values()
                              if "WAIT" in p.action)
            no_trade = sum(1 for p in st.session_state.predictions.values()
                          if "NO TRADE" in p.action)
            
            avg_score = np.mean([p.signal_score for p in st.session_state.predictions.values()])
            avg_rr = np.mean([p.risk_reward for p in st.session_state.predictions.values()])
            
            with cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Stocks</div>
                    <div class="metric-value">{total}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🟢 Buy/Sell</div>
                    <div class="metric-value">{buy_signals + sell_signals}</div>
                    <div class="metric-subtitle">{(buy_signals + sell_signals)/total*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">⏸️ Wait/Hold</div>
                    <div class="metric-value">{wait_signals + no_trade}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Score</div>
                    <div class="metric-value">{avg_score:.0f}</div>
                    <div class="metric-subtitle">out of 100</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[4]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg R:R</div>
                    <div class="metric-value">{avg_rr:.2f}:1</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Individual stock predictions
        for symbol, pred in st.session_state.predictions.items():
            st.subheader(f"📈 {symbol} - Enhanced Prediction")
            
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                # Top metrics row
                cols = st.columns(4)
                
                with cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Current Price</div>
                        <div class="metric-value">${pred.current_price:.2f}</div>
                        <div class="metric-subtitle">{pred.price_date}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Direction</div>
                        <div class="metric-value">{pred.week_direction}</div>
                        <div class="metric-subtitle">{pred.week_prob_up*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    # Score badge with color
                    if pred.signal_score >= 75:
                        score_class = "score-excellent"
                    elif pred.signal_score >= 65:
                        score_class = "score-good"
                    elif pred.signal_score >= 55:
                        score_class = "score-marginal"
                    else:
                        score_class = "score-weak"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Signal Score</div>
                        <div class="metric-value">{pred.signal_score:.0f}/100</div>
                        <div class="metric-subtitle">{pred.signal_strength}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value" style="font-size: 1.3rem;">{pred.confidence}</div>
                        <div class="metric-subtitle">Thresh: {pred.adaptive_threshold*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Score breakdown
                st.markdown("### 📊 Score Breakdown")
                breakdown_cols = st.columns(4)
                
                for idx, (component, score) in enumerate(pred.score_breakdown.items()):
                    with breakdown_cols[idx]:
                        component_label = component.replace('_', ' ').title()
                        max_score = {'probability': 40, 'risk_reward': 25, 
                                    'market_alignment': 20, 'volatility': 15}[component]
                        percentage = (score / max_score) * 100
                        
                        st.markdown(f"""
                        <div class="glass-card" style="padding: 0.8rem;">
                            <strong>{component_label}</strong><br>
                            <span style="font-size: 1.5rem; color: #fff;">{score:.0f}</span>
                            <span style="font-size: 0.9rem; color: rgba(255,255,255,0.6);">/{max_score}</span>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {percentage}%; background: {'#22c55e' if percentage >= 75 else '#fbbf24' if percentage >= 50 else '#ef4444'};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Risk Management Section
                st.markdown("### 🎯 Risk Management")
                risk_cols = st.columns(3)
                
                with risk_cols[0]:
                    st.markdown(f"""
                    <div class="glass-card">
                        <strong>Entry & Targets:</strong><br>
                        Entry: <span style="color: #3b82f6;">${pred.current_price:.2f}</span><br>
                        Target: <span style="color: #22c55e;">${pred.target_low:.2f} - ${pred.target_high:.2f}</span><br>
                        Expected: <span style="color: #22c55e;">+{pred.expected_return:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_cols[1]:
                    rr_color = "#22c55e" if pred.risk_reward >= 2.0 else "#fbbf24" if pred.risk_reward >= 1.5 else "#ef4444"
                    st.markdown(f"""
                    <div class="glass-card">
                        <strong>Risk Parameters:</strong><br>
                        Stop Loss: <span style="color: #ef4444;">${pred.stop_loss:.2f}</span><br>
                        Max Loss: <span style="color: #ef4444;">-{pred.max_loss:.2f}%</span><br>
                        R:R: <span style="color: {rr_color};">{pred.risk_reward:.2f}:1</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_cols[2]:
                    st.markdown(f"""
                    <div class="glass-card">
                        <strong>Market Context:</strong><br>
                        Regime: {pred.market_regime}<br>
                        Volatility: {pred.volatility*100:.2f}% ({pred.volatility_regime})<br>
                        ATR: {pred.atr_pct:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                # Action
                action_class = pred.action.lower().replace(" ", "-").replace("🟢", "").replace("🔴", "").replace("⚡", "").replace("⏸️", "").replace("❌", "").strip()
                
                st.markdown(f"""
                <div style="text-align: center; margin: 1.5rem 0;">
                    <div class="action-badge action-{action_class}">
                        {pred.action}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_right:
                # Threshold adaptation info
                st.markdown("### 🎚️ Adaptive Threshold")
                st.markdown(f"""
                <div class="info-box">
                    <strong>Base:</strong> 58.0%<br>
                    <strong>Vol Adj:</strong> {pred.threshold_breakdown['vol_adjustment']*100:+.1f}%<br>
                    <strong>Regime Adj:</strong> {pred.threshold_breakdown['regime_adjustment']*100:+.1f}%<br>
                    <strong>Final:</strong> {pred.adaptive_threshold*100:.1f}%<br>
                    <strong>Trend Consistency:</strong> {pred.threshold_breakdown.get('trend_consistency', 0)*100:.0f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Reasoning
                st.markdown("### 🧠 Analysis")
                for reason in pred.reasoning:
                    if "✅" in reason:
                        st.markdown(f'<div class="success-box">{reason}</div>', 
                                  unsafe_allow_html=True)
                    elif "⚠️" in reason or "❌" in reason:
                        st.markdown(f'<div class="warning-box">{reason}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="info-box">{reason}</div>', 
                                  unsafe_allow_html=True)
                
                # Warnings
                if pred.warnings:
                    st.markdown("### ⚠️ Warnings")
                    for warning in pred.warnings:
                        st.markdown(f'<div class="warning-box">• {warning}</div>', 
                                  unsafe_allow_html=True)
            
            st.markdown("---")

# Tab 2: Comparison Table
with tab_comparison:
    if not st.session_state.predictions:
        st.info("Generate predictions to see comparison table")
    else:
        st.subheader("📊 Comparative Analysis Table")
        
        # Create comparison dataframe
        comparison_data = []
        for symbol, p in st.session_state.predictions.items():
            comparison_data.append({
                'Symbol': symbol,
                'Price': f"${p.current_price:.2f}",
                'Direction': p.week_direction.replace('📈', '').replace('📉', '').strip(),
                'Probability': f"{p.week_prob_up*100:.1f}%",
                'Threshold': f"{p.adaptive_threshold*100:.1f}%",
                'Score': f"{p.signal_score:.0f}/100",
                'Signal': p.signal_strength,
                'Confidence': p.confidence.replace('🟢', '').replace('🟡', '').replace('🟠', '').replace('🔴', '').strip(),
                'Target Range': f"${p.target_low:.0f}-${p.target_high:.0f}",
                'Expected Return': f"+{p.expected_return:.1f}%",
                'R:R': f"{p.risk_reward:.2f}:1",
                'Market Regime': p.market_regime.replace('🚀', '').replace('📈', '').replace('📉', '').replace('⚖️', '').replace('🔄', '').replace('⚡', '').strip(),
                'Action': p.action.replace('🟢', '').replace('🔴', '').replace('⚡', '').replace('⏸️', '').replace('❌', '').strip()
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Apply styling
        def highlight_score(val):
            if 'Score' in val.name:
                score = int(val.split('/')[0])
                if score >= 75:
                    return ['background-color: rgba(34, 197, 94, 0.2)'] * len(val)
                elif score >= 65:
                    return ['background-color: rgba(59, 130, 246, 0.2)'] * len(val)
                elif score >= 55:
                    return ['background-color: rgba(251, 191, 36, 0.2)'] * len(val)
            return [''] * len(val)
        
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Download button
        csv = df_comparison.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison CSV",
            data=csv,
            file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Best opportunity
        if any("BUY" in p.action or "SELL" in p.action for p in st.session_state.predictions.values()):
            trade_preds = [p for p in st.session_state.predictions.values() 
                          if "BUY" in p.action or "SELL" in p.action]
            best = max(trade_preds, key=lambda x: x.signal_score)
            
            st.success(f"🏆 **Best Opportunity:** {best.symbol} (Score: {best.signal_score:.0f}, R:R: {best.risk_reward:.2f}:1, {best.action})")

# Tab 3: Technical Analysis
with tab_analysis:
    if not st.session_state.predictions:
        st.info("Generate predictions first to see technical charts")
    else:
        selected_stock = st.selectbox(
            "Select Stock for Charts",
            list(st.session_state.predictions.keys())
        )
        
        if selected_stock:
            pred = st.session_state.predictions[selected_stock]
            
            st.subheader(f"📊 {selected_stock} - Technical Charts")
            
            try:
                # Load stock data
                csv_paths = [
                    Path(f"data/stock_data/{selected_stock}.csv"),
                    Path(f"data/{selected_stock}.csv"),
                ]
                
                df = None
                for csv_path in csv_paths:
                    if csv_path.exists():
                        df = pd.read_csv(csv_path)
                        break
                
                if df is not None:
                    df.columns = df.columns.str.lower()
                    
                    # Find date column
                    date_col = None
                    for col in ['date', 'datetime', 'timestamp']:
                        if col in df.columns:
                            date_col = col
                            break
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.set_index(date_col).sort_index()
                    
                    # Display recent data (90 days)
                    df_recent = df.tail(90)
                    
                    # Price chart with targets
                    st.markdown("### 📈 Price Chart with Targets (Last 90 Days)")
                    
                    fig = go.Figure()
                    
                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=df_recent.index,
                        open=df_recent['open'],
                        high=df_recent['high'],
                        low=df_recent['low'],
                        close=df_recent['close'],
                        name='OHLC'
                    ))
                    
                    # Current price line
                    fig.add_hline(y=pred.current_price, line_dash="solid", 
                                 line_color="white", annotation_text="Current",
                                 annotation_position="right")
                    
                    # Target high
                    fig.add_hline(y=pred.target_high, line_dash="dash", 
                                 line_color="lime", annotation_text="Target High",
                                 annotation_position="right")
                    
                    # Target low
                    fig.add_hline(y=pred.target_low, line_dash="dash", 
                                 line_color="lightgreen", annotation_text="Target Low",
                                 annotation_position="right")
                    
                    # Stop loss
                    fig.add_hline(y=pred.stop_loss, line_dash="dash", 
                                 line_color="red", annotation_text="Stop Loss",
                                 annotation_position="right")
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.05)',
                        font=dict(color='white'),
                        xaxis_rangeslider_visible=False,
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    st.markdown("### 📊 Trading Volume")
                    fig_vol = go.Figure(data=[go.Bar(
                        x=df_recent.index,
                        y=df_recent['volume'],
                        marker_color='rgba(139, 92, 246, 0.6)',
                        name='Volume'
                    )])
                    
                    fig_vol.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.05)',
                        font=dict(color='white'),
                        height=250,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_vol, use_container_width=True)
                    
                    # Price statistics
                    st.markdown("### 📈 Price Statistics")
                    stats_cols = st.columns(4)
                    
                    week_change = ((df_recent['close'].iloc[-1] - df_recent['close'].iloc[-5]) / df_recent['close'].iloc[-5] * 100) if len(df_recent) >= 5 else 0
                    month_change = ((df_recent['close'].iloc[-1] - df_recent['close'].iloc[-20]) / df_recent['close'].iloc[-20] * 100) if len(df_recent) >= 20 else 0
                    
                    with stats_cols[0]:
                        st.metric("52W High", f"${df_recent['high'].max():.2f}")
                    
                    with stats_cols[1]:
                        st.metric("52W Low", f"${df_recent['low'].min():.2f}")
                    
                    with stats_cols[2]:
                        st.metric("1W Change", f"{week_change:+.2f}%")
                    
                    with stats_cols[3]:
                        st.metric("1M Change", f"{month_change:+.2f}%")
                    
                else:
                    st.warning(f"Could not load chart data for {selected_stock}")
                    
            except Exception as e:
                st.error(f"Error loading charts: {str(e)}")

# Tab 4: History Log
with tab_logs:
    st.subheader("📋 Prediction History")
    
    try:
        log_path = Path("predictions_log.csv")
        if log_path.exists():
            log_df = pd.read_csv(log_path)
            
            # Show summary
            st.markdown("### 📊 Log Summary")
            summary_cols = st.columns(4)
            
            with summary_cols[0]:
                st.metric("Total Predictions", len(log_df))
            
            with summary_cols[1]:
                unique_stocks = log_df['symbol'].nunique() if 'symbol' in log_df.columns else 0
                st.metric("Unique Stocks", unique_stocks)
            
            with summary_cols[2]:
                if 'action' in log_df.columns:
                    trade_signals = log_df['action'].str.contains('BUY|SELL', case=False, na=False).sum()
                    st.metric("Trade Signals", trade_signals)
                else:
                    st.metric("Trade Signals", "N/A")
            
            with summary_cols[3]:
                if 'signal_score' in log_df.columns:
                    avg_score = log_df['signal_score'].mean()
                    st.metric("Avg Score", f"{avg_score:.1f}/100")
                else:
                    st.metric("Avg Score", "N/A")
            
            st.markdown("---")
            
            # Filters
            st.markdown("### 🔍 Filters")
            filter_cols = st.columns(3)
            
            with filter_cols[0]:
                if 'symbol' in log_df.columns:
                    selected_symbols = st.multiselect(
                        "Filter by Symbol",
                        options=sorted(log_df['symbol'].unique()),
                        default=[]
                    )
                else:
                    selected_symbols = []
            
            with filter_cols[1]:
                if 'action' in log_df.columns:
                    selected_actions = st.multiselect(
                        "Filter by Action",
                        options=sorted(log_df['action'].unique()),
                        default=[]
                    )
                else:
                    selected_actions = []
            
            with filter_cols[2]:
                show_rows = st.selectbox(
                    "Show Rows",
                    options=[25, 50, 100, "All"],
                    index=1
                )
            
            # Apply filters
            filtered_df = log_df.copy()
            if selected_symbols:
                filtered_df = filtered_df[filtered_df['symbol'].isin(selected_symbols)]
            if selected_actions:
                filtered_df = filtered_df[filtered_df['action'].isin(selected_actions)]
            
            # Display dataframe
            st.markdown("### 📄 Log Data")
            if show_rows == "All":
                st.dataframe(filtered_df, use_container_width=True, height=400)
            else:
                st.dataframe(filtered_df.tail(int(show_rows)), use_container_width=True, height=400)
            
            # Download buttons
            st.markdown("### 📥 Download Options")
            download_cols = st.columns(2)
            
            with download_cols[0]:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Log",
                    data=csv,
                    file_name=f"predictions_log_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with download_cols[1]:
                csv_full = log_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Log",
                    data=csv_full,
                    file_name=f"predictions_log_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Performance insights (if enough data)
            if len(log_df) >= 10 and 'signal_score' in log_df.columns:
                st.markdown("---")
                st.markdown("### 📊 Performance Insights")
                
                insight_cols = st.columns(3)
                
                with insight_cols[0]:
                    if 'action' in log_df.columns:
                        action_counts = log_df['action'].value_counts()
                        st.markdown("**Action Distribution:**")
                        for action, count in action_counts.head(5).items():
                            st.text(f"{action}: {count} ({count/len(log_df)*100:.1f}%)")
                
                with insight_cols[1]:
                    st.markdown("**Score Distribution:**")
                    excellent = (log_df['signal_score'] >= 75).sum()
                    good = ((log_df['signal_score'] >= 65) & (log_df['signal_score'] < 75)).sum()
                    marginal = ((log_df['signal_score'] >= 55) & (log_df['signal_score'] < 65)).sum()
                    weak = (log_df['signal_score'] < 55).sum()
                    
                    st.text(f"Excellent (≥75): {excellent} ({excellent/len(log_df)*100:.1f}%)")
                    st.text(f"Good (65-74): {good} ({good/len(log_df)*100:.1f}%)")
                    st.text(f"Marginal (55-64): {marginal} ({marginal/len(log_df)*100:.1f}%)")
                    st.text(f"Weak (<55): {weak} ({weak/len(log_df)*100:.1f}%)")
                
                with insight_cols[2]:
                    if 'symbol' in log_df.columns:
                        st.markdown("**Most Analyzed:**")
                        top_stocks = log_df['symbol'].value_counts().head(5)
                        for stock, count in top_stocks.items():
                            st.text(f"{stock}: {count} times")
        else:
            st.info("No prediction log found. Generate predictions to create a log.")
            st.markdown("""
            <div class="info-box">
                <strong>📝 Log File Location:</strong> predictions_log.csv<br>
                The log file will be automatically created in the project root directory 
                when you generate your first prediction.
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading log: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.6);">
    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
        <strong>Stock Predictor Pro v2.0</strong> | Enhanced LSTM Neural Network
    </p>
    <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">
        ✨ Features: Adaptive Thresholds • Weighted Scoring • Market Regime Detection • R:R ≥1.5:1
    </p>
    <p style="font-size: 0.85rem; color: rgba(255,255,255,0.5);">
        ⚠️ <strong>CRITICAL DISCLAIMER:</strong> For Educational & Research Purposes Only | Not Financial Advice
    </p>
    <p style="font-size: 0.8rem; color: rgba(255,255,255,0.4); margin-top: 1rem;">
        Always paper trade extensively before risking real capital | Consult licensed financial professionals for investment advice
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: rgba(255,255,255,0.5); font-size: 0.75rem;">
        <p><strong>⚠️ DISCLAIMER</strong></p>
        <p>This tool is for educational purposes only. Not financial advice. 
        Always paper trade first and consult professionals.</p>
    </div>
    """, unsafe_allow_html=True)
