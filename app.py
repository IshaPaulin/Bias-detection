"""
FAIRNESS AUDITOR DASHBOARD - Google M3 Glass Edition
=====================================================
The user-facing interface for the entire system, optimized for the Google Solution Challenge.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from dotenv import load_dotenv
from proxy_detector import ProxyDetector
from counterfactual_tester import CounterfactualTester
from fairness_metrics import get_fairness_metrics
from bias_narrator import BiasNarrator
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Fairness Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Google M3 + Glassmorphism
def load_css():
    st.markdown("""
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    /* Fallback to Google Sans if available locally */
    @font-face {
      font-family: 'Google Sans';
      src: local('Google Sans'), local('GoogleSans-Regular');
    }

    /* Global Typography & Palette */
    html, body, [class*="css"]  {
        font-family: 'Google Sans', 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* Deep Space Dark Background */
    .stApp {
        background-color: #1a1c1e !important;
        color: #f8fafc;
    }
    
    /* Sidebar with G-Gradient Glow */
    [data-testid="stSidebar"] {
        background: #212429 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 150px;
        background: linear-gradient(135deg, rgba(66, 133, 244, 0.4), rgba(168, 85, 247, 0.4));
        filter: blur(40px);
        z-index: -1;
    }
    
    /* Main Title */
    .main-title {
        color: #fff;
        font-size: 3.5rem !important;
        margin-bottom: 0rem;
    }
    .main-title span {
        color: #4285F4; /* Google Blue */
    }
    
    /* Glassmorphism Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #4285F4;
    }
    
    /* Metric Values - Light Blue */
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem !important;
        font-weight: 800;
        color: #D2E3FC !important;
    }
    
    /* M3 Pill-Shaped Buttons */
    .stButton > button {
        background: #8ab4f8 !important;
        color: #1a1c1e !important;
        font-family: 'Google Sans', 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 100px !important;
        border: none !important;
        padding: 0.6rem 2.5rem !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(138, 180, 248, 0.4);
        background: #a8c7fa !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding: 10px 15px;
        font-family: 'Google Sans', 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: #9aa0a6;
    }
    .stTabs [aria-selected="true"] {
        color: #8ab4f8 !important;
        border-bottom: 3px solid #8ab4f8 !important;
    }
    
    /* Material Alert Box for Victim Narrative */
    .material-alert {
        background: #212429;
        border-left: 6px solid #D2E3FC;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 16px;
        box-shadow: inset 0 0 20px rgba(210, 227, 252, 0.05), 0 4px 15px rgba(0,0,0,0.2);
    }
    .material-alert h4 {
        color: #D2E3FC;
        margin-top: 0;
        margin-bottom: 8px;
    }
    .material-alert p {
        color: #e8eaed;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .material-alert p:last-child {
        margin-bottom: 0;
    }
    
    /* Expanders & DataFrames */
    .streamlit-expanderHeader {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
    }
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Tab Headers */
    .tab-header {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        color: #f8fafc;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Title and description
st.markdown('<h1 class="main-title">⚖️ AI Fairness <span>Auditor</span></h1>', unsafe_allow_html=True)
st.markdown("""
<div style='font-size: 1.2rem; color: #9aa0a6; margin-bottom: 2rem;'>
    <strong>Detect algorithmic bias before it causes harm. Built for the Google Solution Challenge.</strong>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.markdown('<h2 style="margin-top: 0; color: #fff;">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    use_sample_data = st.checkbox("Use sample biased loan data", value=True)
    
    if not use_sample_data:
        uploaded_file = st.file_uploader(
            "Upload your decision dataset (CSV)",
            type=['csv'],
            help="Should include applicant features and final decisions"
        )
    else:
        uploaded_file = None
        
    st.markdown("---")
    
    with st.expander("📖 How to use", expanded=True):
        st.markdown("""
        1. **Upload data** with historical decisions.
        2. **Select outcome column** (e.g., approved/rejected).
        3. **Choose sensitive attributes** to audit.
        4. **Run the audit** and review results.
        5. **Generate Report** to see the Agentic analysis.
        """)

# Tabs Layout
tab1, tab2, tab3 = st.tabs(["📂 Data Ingestion", "📈 Statistical Evidence", "🕵️ Investigation Report"])

with tab1:
    st.markdown('<div class="tab-header">Data Setup & Audit</div>', unsafe_allow_html=True)
    df = None
    if use_sample_data or uploaded_file is not None:
        
        # Load data
        if use_sample_data:
            df = pd.read_csv('data/sample_loan_data.csv')
            st.toast("Loaded sample biased loan dataset!", icon="✅")
        else:
            df = pd.read_csv(uploaded_file)
            st.toast(f"Loaded {len(df)} records!", icon="✅")
        
        # Show data preview
        with st.expander("📋 View Data Structure & Preview", expanded=False):
            st.dataframe(df.head(10))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                st.metric("Numerical Features", len(numerical_cols))
            with col3:
                categorical_cols = df.select_dtypes(include=['object']).columns
                st.metric("Categorical Features", len(categorical_cols))
        
        # Configuration
        st.markdown("### 🎯 Audit Target")
        
        col1, col2 = st.columns(2)
        
        with col1:
            outcome_col = st.selectbox(
                "Select the outcome column",
                options=df.columns,
                index=list(df.columns).index('approved') if 'approved' in df.columns else 0,
                help="This is the decision your AI made (1=approved, 0=rejected)"
            )
        
        with col2:
            sensitive_attr = st.selectbox(
                "Select known sensitive attribute (optional)",
                options=['None'] + list(df.columns),
                index=list(df.columns).index('zip_code') + 1 if 'zip_code' in df.columns else 0,
                help="If you know which attribute should NOT matter (e.g., zip_code), we'll focus on it"
            )
            if sensitive_attr == 'None':
                sensitive_attr = None
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Run audit button
        if st.button("🔬 Run Deep Fairness Audit", use_container_width=True):
            
            with st.status("Performing Deep Fairness Audit...", expanded=True) as status:
                
                st.write("🔍 Step 1: Detecting proxy variables (Mutual Information & Pearson)...")
                detector = ProxyDetector(suspicious_threshold=0.25)
                proxies = detector.detect_proxies(df, outcome_col=outcome_col, sensitive_attr=sensitive_attr)
                time.sleep(0.5) # Slight delay for UI UX
                
                st.write("📊 Step 2: Computing Disparate Impact Ratio & Demographic Parity...")
                metrics_list = []
                for proxy in list(proxies.keys())[:5]:
                    metrics = get_fairness_metrics(df, outcome_col, proxy)
                    metrics_list.append(metrics)
                time.sleep(0.5)
                
                st.write("🧪 Step 3: Training model & testing counterfactual 'Twins'...")
                tester = CounterfactualTester()
                tester.train_biased_model(df, outcome_col=outcome_col)
                
                variables_to_test = list(proxies.keys())[:5] if len(proxies) > 0 else ['zip_code', 'browser']
                
                flagged_cases = tester.test_counterfactual_fairness(
                    df, 
                    outcome_col=outcome_col,
                    variables_to_test=variables_to_test,
                    sample_size=100
                )
                
                status.update(label="Audit Complete!", state="complete", expanded=False)
            
            # Save to session state
            st.session_state['audit_run'] = True
            st.session_state['df'] = df
            st.session_state['outcome_col'] = outcome_col
            st.session_state['proxies'] = proxies
            st.session_state['flagged_cases'] = flagged_cases
            st.session_state['detector'] = detector
            st.session_state['metrics_list'] = metrics_list
            
            st.toast("Audit finalized! Check the Evidence tab.", icon="🎉")
    else:
        st.info("👆 Select 'Use sample data' or upload your own CSV in the sidebar to begin.")

with tab2:
    if st.session_state.get('audit_run'):
        st.markdown('<div class="tab-header">Mathematical Proof of Bias</div>', unsafe_allow_html=True)
        
        df = st.session_state['df']
        outcome_col = st.session_state['outcome_col']
        proxies = st.session_state['proxies']
        flagged_cases = st.session_state['flagged_cases']
        detector = st.session_state['detector']
        
        # Calculate fairness score
        total_rejected = len(df[df[outcome_col] == 0])
        biased_rejections = len(flagged_cases) if len(flagged_cases) > 0 else 0
        fairness_score = max(0, 100 - (biased_rejections / total_rejected * 100)) if total_rejected > 0 else 100
        
        # Top Row Metrics & Gauge
        col1, col2, col3 = st.columns([1, 1, 1.5])
        with col1:
            st.metric("Proxy Variables Found", len(proxies))
        with col2:
            st.metric("Biased Decisions Detected", biased_rejections)
        with col3:
            # Render Gauge Chart for Fairness Score
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = fairness_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Fairness Score", 'font': {'color': '#f8fafc', 'family': 'Outfit'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#34A853" if fairness_score > 80 else ("#FBBC04" if fairness_score > 60 else "#EA4335")},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, 60], 'color': 'rgba(234, 67, 53, 0.2)'},
                        {'range': [60, 80], 'color': 'rgba(251, 188, 4, 0.2)'},
                        {'range': [80, 100], 'color': 'rgba(52, 168, 83, 0.2)'}],
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # Proxy variables section (Horizontal Bar Chart)
        if len(proxies) > 0:
            st.markdown("### 🚨 Proxy Risk Assessment")
            
            proxy_df = pd.DataFrame([{'Variable': k, 'Suspiciousness Score': v} for k, v in list(proxies.items())[:10]])
            
            # Dark theme plotly chart - Google Colors
            fig = px.bar(proxy_df, x='Suspiciousness Score', y='Variable', orientation='h', 
                         color='Suspiciousness Score', color_continuous_scale=['#4285F4', '#EA4335'])
            fig.update_layout(
                template='plotly_dark',
                height=300, 
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8eaed', family='Inter'),
                xaxis_title="Risk Score",
                yaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)
                    
        # Counterfactual cases
        if len(flagged_cases) > 0:
            st.markdown("### ⚖️ Counterfactual 'Flip' Evidence")
            st.markdown("These applicants were **rejected**, but would have been **approved** if only a proxy variable changed.")
            st.dataframe(flagged_cases[['applicant_id', 'variable_changed', 'original_value', 'counterfactual_value', 'credit_score', 'income']].head(20), use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            csv = flagged_cases.to_csv(index=False)
            st.download_button("📥 Download Audit CSV", data=csv, file_name="fairness_audit_report.csv", mime="text/csv")
        else:
            st.success("✅ No clear counterfactual bias detected in this dataset!")
    else:
        st.info("Please run the audit in Tab 1 first.")

with tab3:
    if st.session_state.get('audit_run'):
        st.markdown('<div class="tab-header">Agentic Regulatory Report</div>', unsafe_allow_html=True)
        st.markdown("Our **AI Agents** (Powered by Gemini) analyze the raw mathematical findings to generate real-world narratives of systemic harm.")
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if not gemini_key:
            st.error("🔒 **API Key Missing!** Please create a `.env` file in the root directory and add `GEMINI_API_KEY` to unlock Agentic Reporting.", icon="🚨")
        else:
            if st.button("✨ Generate AI Narrative Report", use_container_width=True):
                with st.spinner("Agents are analyzing the statistical evidence..."):
                    proxies = st.session_state['proxies']
                    flagged_cases = st.session_state['flagged_cases']
                    metrics_list = st.session_state['metrics_list']
                    
                    narrator = BiasNarrator(gemini_api_key=gemini_key)
                    
                    st.markdown("---")
                    st.markdown('<h3><span style="color: #4285F4;">Agent A:</span> The Inspector (Socioeconomic Analysis)</h3>', unsafe_allow_html=True)
                    
                    inspector_analysis = narrator.analyze_proxies_with_gemini(proxies, metrics_list)
                    
                    # Material Alert Box for Inspector
                    st.markdown(f"""
                    <div class="material-alert">
                        <p>{inspector_analysis}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<h3><span style="color: #4285F4;">Agent B:</span> The Explainer (Victim Narrative)</h3>', unsafe_allow_html=True)
                    
                    gemini_json_str = narrator.generate_victim_narrative_with_gemini(flagged_cases, metrics_list)
                    
                    try:
                        cleaned_str = gemini_json_str.strip()
                        if cleaned_str.startswith("```json"):
                            cleaned_str = cleaned_str[7:]
                        if cleaned_str.startswith("```"):
                            cleaned_str = cleaned_str[3:]
                        if cleaned_str.endswith("```"):
                            cleaned_str = cleaned_str[:-3]
                            
                        narrative_data = json.loads(cleaned_str.strip())
                        
                        if "error" in narrative_data:
                            st.error(narrative_data["error"], icon="❌")
                        elif "message" in narrative_data:
                            st.info(narrative_data["message"], icon="ℹ️")
                        else:
                            # Material Alert Box for Victim Narrative
                            st.markdown(f"""
                            <div class="material-alert">
                                <h4>🧑 The Victim Profile</h4>
                                <p>{narrative_data.get("victim_profile", "N/A")}</p>
                                <h4>⚖️ The Injustice</h4>
                                <p>{narrative_data.get("the_injustice", "N/A")}</p>
                                <h4>🌍 Systemic Impact</h4>
                                <p>{narrative_data.get("systemic_impact", "N/A")}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        st.toast("Narrative Report Generated!", icon="📑")
                            
                    except json.JSONDecodeError:
                        st.error("Failed to parse Gemini's response as JSON. Raw output:", icon="⚠️")
                        st.code(gemini_json_str)
    else:
        st.info("Please run the audit in Tab 1 first.")

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #9aa0a6; font-size: 0.9rem;'><p>Google Solution Challenge • AI Fairness Auditor • Built with Streamlit & Gemini</p></div>", unsafe_allow_html=True)
