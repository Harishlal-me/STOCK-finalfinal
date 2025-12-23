import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta


ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))


from config import Config
from src.data_loader import load_stock_data
from predict import ui_predict_for_symbol  # Import from predict.py instead
if "prediction" not in st.session_state:
    st.session_state.prediction = None


st.set_page_config(
    page_title="AI Stock Oracle Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# iOS 26 Glassmorphism CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .hero-ios {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 300;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-tile {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
    }
    
    .rec-chip {
        display: inline-block;
        padding: 0.8rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .rec-buy {
        background: rgba(34, 197, 94, 0.3);
        color: #fff;
    }
    
    .rec-sell {
        background: rgba(239, 68, 68, 0.3);
        color: #fff;
    }
    
    .rec-hold {
        background: rgba(251, 191, 36, 0.3);
        color: #fff;
    }
    
    .chart-glass {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2);
        color: #fff;
    }
    
    .stButton button {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        color: #fff;
        font-weight: 600;
        padding: 0.8rem 2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)


# Hero
st.markdown("""
<div class="hero-ios">
    <div class="hero-title">AI Stock Oracle Pro</div>
    <div class="hero-subtitle">
        Multi-task LSTM • 60-day sequences • Tomorrow & Weekly predictions
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("### 📈 Stock Selection")
    ticker = st.selectbox("Symbol", Config.SUPPORTED_STOCKS, index=0)
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.markdown("Select a stock and click **Generate AI Prediction** to see signals.")


# Tabs
tab_live, tab_data, tab_about = st.tabs(
    ["🔮 Live Prediction", "📊 Market Data", "ℹ️ About"]
)


with tab_live:
    left, right = st.columns([2.2, 1.8])
    
    with left:
        st.subheader(f"🎯 Predicting {ticker}")
        
        # ✅ FIX 2: Button ONLY stores prediction in session_state
        if st.button("🚀 Generate AI Prediction", use_container_width=True):
            with st.spinner("Running LSTM model..."):
                st.session_state.prediction = ui_predict_for_symbol(ticker)
                st.success("✅ Prediction generated successfully!")
        
        # ✅ FIX 3: ALL METRICS UI MOVED OUTSIDE BUTTON - MOST IMPORTANT!
        if st.session_state.prediction is not None:
            metrics = st.session_state.prediction

            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">${metrics.current_price:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Tomorrow</div>
                    <div class="metric-value">{metrics.tom_direction}</div>
                    <div class="metric-label">{metrics.p_tom_up:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">This Week</div>
                    <div class="metric-value">{metrics.week_direction}</div>
                    <div class="metric-label">{metrics.p_week_up:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendation
            rec_class = f"rec-{metrics.action.lower()}"
            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0;">
                <div class="rec-chip {rec_class}">
                    🎯 {metrics.action} SIGNAL
                </div>
                <div style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">
                    Signal Strength: {metrics.signal_strength}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probabilities chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Probability',
                x=['Tomorrow', 'Week'],
                y=[metrics.p_tom_up * 100, metrics.p_week_up * 100],
                marker_color=['rgba(99, 102, 241, 0.8)', 'rgba(139, 92, 246, 0.8)'],
                text=[f"{metrics.p_tom_up:.1%}", f"{metrics.p_week_up:.1%}"],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Probability of Price Going UP",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(range=[0, 100], title="Probability (%)"),
                showlegend=False,
                height=300
            )
            
            st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Click **Generate AI Prediction** to see results")
    
    with right:
        st.markdown("""
        <div class="glass-card">
            <h3>🧠 How It Works</h3>
            <p style="color: rgba(255,255,255,0.8); line-height: 1.6;">
            Our LSTM neural network analyzes 60 days of market data including:
            </p>
            <ul style="color: rgba(255,255,255,0.7);">
                <li>Price movements (OHLC)</li>
                <li>Technical indicators (RSI, MACD, Bollinger Bands)</li>
                <li>Volume patterns</li>
                <li>Momentum signals</li>
            </ul>
            <hr style="border-color: rgba(255,255,255,0.2);">
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            <strong>Model Accuracy:</strong><br>
            Tomorrow: 60%<br>
            Week: 67%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Signal Thresholds</h3>
            <ul style="color: rgba(255,255,255,0.7); line-height: 1.8;">
                <li><strong>BUY:</strong> P(UP) ≥ 55%</li>
                <li><strong>SELL:</strong> P(UP) ≤ 45%</li>
                <li><strong>HOLD:</strong> 45% < P(UP) < 55%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


with tab_data:
    st.subheader(f"📈 {ticker} Historical Data")
    
    try:
        df = load_stock_data(ticker)
        
        if df is not None and not df.empty:
            # Date range selector
            col1, col2 = st.columns(2)
            with col1:
                days_back = st.selectbox("Time Period", [30, 60, 90, 180, 365], index=2)
            
            df_display = df.tail(days_back)
            
            # Candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df_display.index,
                open=df_display['Open'],
                high=df_display['High'],
                low=df_display['Low'],
                close=df_display['Close'],
                name='OHLC'
            )])
            
            fig.update_layout(
                title=f"{ticker} Price Movement (Last {days_back} Days)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)',
                font=dict(color='white'),
                xaxis_rangeslider_visible=False,
                height=400
            )
            
            st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Volume chart
            fig_vol = go.Figure(data=[go.Bar(
                x=df_display.index,
                y=df_display['Volume'],
                marker_color='rgba(139, 92, 246, 0.6)',
                name='Volume'
            )])
            
            fig_vol.update_layout(
                title="Trading Volume",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)',
                font=dict(color='white'),
                height=250
            )
            
            st.markdown('<div class="chart-glass">', unsafe_allow_html=True)
            st.plotly_chart(fig_vol, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Data table
            with st.expander("📄 View Raw Data"):
                st.dataframe(df_display.tail(20), use_container_width=True)
        else:
            st.warning("No data available for this symbol")
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


with tab_about:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🏗️ Architecture</h3>
            <p style="color: rgba(255,255,255,0.8); line-height: 1.6;">
            <strong>Multi-Task LSTM Model</strong><br><br>
            • <strong>Input:</strong> 60-day sequences of market data<br>
            • <strong>Features:</strong> 20+ technical indicators<br>
            • <strong>Hidden Layers:</strong> 2x LSTM (128, 64 units)<br>
            • <strong>Outputs:</strong> 4 predictions<br>
               &nbsp;&nbsp;- Tomorrow direction (binary)<br>
               &nbsp;&nbsp;- Week direction (binary)<br>
               &nbsp;&nbsp;- Tomorrow return (regression)<br>
               &nbsp;&nbsp;- Week return (regression)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
            <h3>📚 Training Data</h3>
            <ul style="color: rgba(255,255,255,0.8);">
                <li>15+ years historical data</li>
                <li>Daily OHLCV prices</li>
                <li>Technical indicators computed on-the-fly</li>
                <li>Train/validation split: 80/20</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>⚙️ Technical Indicators</h3>
            <ul style="color: rgba(255,255,255,0.8); line-height: 1.8;">
                <li>RSI (Relative Strength Index)</li>
                <li>MACD (Moving Average Convergence Divergence)</li>
                <li>Bollinger Bands</li>
                <li>Moving Averages (5, 20, 50 day)</li>
                <li>Volume indicators</li>
                <li>Momentum oscillators</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
            <h3>⚠️ Disclaimer</h3>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; line-height: 1.6;">
            This tool is for <strong>educational purposes only</strong>. 
            Stock market predictions are inherently uncertain. 
            Past performance does not guarantee future results. 
            Always consult with a financial advisor before making investment decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; color: rgba(255,255,255,0.5);">
    <p>AI Stock Oracle Pro v1.0 | Powered by TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)
