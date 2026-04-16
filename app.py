"""
FAIRNESS AUDITOR DASHBOARD
===========================
The user-facing interface for the entire system.

USER WORKFLOW:
1. Upload CSV of historical decisions
2. Select which column is the outcome (approved/rejected, hired/not hired)
3. Click "Run Audit"
4. Get a fairness report with:
   - Overall fairness score
   - List of proxy variables
   - Specific cases where bias affected outcomes
   - Downloadable evidence for regulators

HOW THE PIECES FIT TOGETHER:
Dashboard → ProxyDetector (finds suspicious vars)
         → CounterfactualTester (proves causation)
         → Report Generator (explains to humans)
"""

import streamlit as st
import pandas as pd
import numpy as np
from proxy_detector import ProxyDetector
from counterfactual_tester import CounterfactualTester
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI Fairness Auditor",
    page_icon="⚖️",
    layout="wide"
)

# Title and description
st.title("⚖️ AI Fairness Auditor")
st.markdown("""
**Detect algorithmic bias before it causes harm.**

This tool analyzes your AI's decisions to find hidden discrimination patterns.
Upload your historical decision data (loans, hiring, admissions, etc.) and we'll:
- 🔍 Find proxy variables that encode bias
- 🧪 Test if changing those variables flips outcomes
- 📊 Generate evidence for auditors and regulators
""")

st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
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
    st.markdown("""
    ### 📖 How to use
    
    1. **Upload data** with historical decisions
    2. **Select outcome column** (approved/rejected)
    3. **Choose sensitive attributes** to audit
    4. **Run the audit** and review results
    
    ### 💡 What makes a good dataset?
    
    - At least 500 records
    - Mix of approved/rejected outcomes
    - Multiple applicant features
    - Historical decisions (not predictions)
    """)

# Main content
if use_sample_data or uploaded_file is not None:
    
    # Load data
    if use_sample_data:
        df = pd.read_csv('data/sample_loan_data.csv')
        st.success("✅ Loaded sample biased loan dataset (1000 applications)")
    else:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} records")
    
    # Show data preview
    with st.expander("📋 Data Preview"):
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
    st.header("🎯 Audit Configuration")
    
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
    
    # Run audit button
    if st.button("🔬 Run Fairness Audit", type="primary"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Detect proxies
        status_text.text("🔍 Step 1/3: Detecting proxy variables...")
        progress_bar.progress(33)
        
        detector = ProxyDetector(suspicious_threshold=0.25)
        proxies = detector.detect_proxies(df, outcome_col=outcome_col, sensitive_attr=sensitive_attr)
        
        # Step 2: Train model and run counterfactual tests
        status_text.text("🧪 Step 2/3: Training model and testing counterfactuals...")
        progress_bar.progress(66)
        
        tester = CounterfactualTester()
        tester.train_biased_model(df, outcome_col=outcome_col)
        
        variables_to_test = list(proxies.keys())[:5] if len(proxies) > 0 else ['zip_code', 'browser']
        
        flagged_cases = tester.test_counterfactual_fairness(
            df, 
            outcome_col=outcome_col,
            variables_to_test=variables_to_test,
            sample_size=100
        )
        
        # Step 3: Generate report
        status_text.text("📊 Step 3/3: Generating fairness report...")
        progress_bar.progress(100)
        
        status_text.empty()
        progress_bar.empty()
        
        # Display results
        st.markdown("---")
        st.header("📊 Audit Results")
        
        # Calculate fairness score
        total_rejected = len(df[df[outcome_col] == 0])
        biased_rejections = len(flagged_cases) if len(flagged_cases) > 0 else 0
        fairness_score = max(0, 100 - (biased_rejections / total_rejected * 100)) if total_rejected > 0 else 100
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Fairness Score",
                f"{fairness_score:.0f}/100",
                delta=f"{fairness_score - 50:.0f} vs baseline",
                delta_color="normal" if fairness_score > 70 else "inverse"
            )
        
        with col2:
            st.metric(
                "Proxy Variables Found",
                len(proxies),
                help="Variables that correlate suspiciously with outcomes"
            )
        
        with col3:
            st.metric(
                "Biased Decisions",
                biased_rejections,
                help="Cases where changing a proxy variable would flip the outcome"
            )
        
        # Proxy variables section
        if len(proxies) > 0:
            st.subheader("🚨 Suspicious Proxy Variables")
            st.markdown("""
            These variables show unusually strong correlations with outcomes.
            They may be encoding protected characteristics like race, gender, or socioeconomic status.
            """)
            
            # Create visualization
            proxy_df = pd.DataFrame([
                {'Variable': k, 'Suspiciousness Score': v}
                for k, v in list(proxies.items())[:10]
            ])
            
            fig = px.bar(
                proxy_df,
                x='Suspiciousness Score',
                y='Variable',
                orientation='h',
                color='Suspiciousness Score',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed explanations
            with st.expander("📖 Detailed Proxy Analysis"):
                for var in list(proxies.keys())[:5]:
                    st.markdown(detector.get_proxy_explanation(var, df, outcome_col))
        
        # Counterfactual cases
        if len(flagged_cases) > 0:
            st.subheader("⚖️ Counterfactual Bias Evidence")
            st.markdown("""
            These applicants were **rejected**, but would have been **approved** if only
            a proxy variable changed. Their qualifications stayed identical.
            """)
            
            # Show sample cases
            st.dataframe(
                flagged_cases[['applicant_id', 'variable_changed', 'original_value', 
                              'counterfactual_value', 'credit_score', 'income']].head(20),
                use_container_width=True
            )
            
            # Variable breakdown
            st.subheader("📊 Which Variables Cause Bias?")
            flip_counts = flagged_cases['variable_changed'].value_counts()
            
            fig = px.pie(
                values=flip_counts.values,
                names=flip_counts.index,
                title="Distribution of Biased Decisions by Variable"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download report
            st.markdown("### 📥 Export Results")
            csv = flagged_cases.to_csv(index=False)
            st.download_button(
                label="Download Full Bias Report (CSV)",
                data=csv,
                file_name="fairness_audit_report.csv",
                mime="text/csv"
            )
        
        else:
            st.success("✅ No clear counterfactual bias detected in this dataset!")
        
        # Recommendations
        st.markdown("---")
        st.header("💡 Recommendations")
        
        if fairness_score < 50:
            st.error("""
            **⚠️ CRITICAL: Significant bias detected**
            
            Immediate actions:
            1. Pause automated decision-making
            2. Review flagged cases manually
            3. Retrain model without proxy variables
            4. Consult legal/compliance team
            """)
        elif fairness_score < 70:
            st.warning("""
            **⚠️ Moderate bias detected**
            
            Recommended actions:
            1. Add human review for borderline cases
            2. Monitor proxy variables closely
            3. Consider fairness-aware training methods
            """)
        else:
            st.success("""
            **✅ Model appears reasonably fair**
            
            Best practices:
            1. Continue monitoring with regular audits
            2. Document your fairness testing process
            3. Stay alert for emerging proxy variables
            """)

else:
    # Landing state
    st.info("👆 Select 'Use sample data' or upload your own CSV to begin")
    
    st.markdown("### 🎯 What this tool does")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🔍 Proxy Detection
        Identifies variables that shouldn't matter but do (zip codes, names, browsers)
        """)
    
    with col2:
        st.markdown("""
        #### 🧪 Counterfactual Testing
        Proves causation by showing "what-if" scenarios
        """)
    
    with col3:
        st.markdown("""
        #### 📊 Clear Reporting
        Generates evidence for auditors, lawyers, regulators
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built to expose algorithmic bias • Not legal advice • Consult compliance experts</p>
</div>
""", unsafe_allow_html=True)
