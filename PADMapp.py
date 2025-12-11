# PADMapp.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="PADM - DIC Risk Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern design
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    /* Risk indicators */
    .risk-low {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 8px solid #28a745;
        color: #155724;
        margin: 10px 0;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 8px solid #ffc107;
        color: #856404;
        margin: 10px 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 8px solid #dc3545;
        color: #721c24;
        margin: 10px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Subheader styling */
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    /* Progress bar custom */
    .risk-progress {
        height: 25px;
        border-radius: 12px;
        margin: 20px 0;
        overflow: hidden;
    }
    
    .risk-progress .low {
        background: linear-gradient(90deg, #d4fc79, #96e6a1);
        height: 100%;
    }
    
    .risk-progress .medium {
        background: linear-gradient(90deg, #f6d365, #fda085);
        height: 100%;
    }
    
    .risk-progress .high {
        background: linear-gradient(90deg, #f093fb, #f5576c);
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained PADM model"""
    try:
        model_info = joblib.load('PADM_model.pkl')
        return model_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_risk(model_info, input_data):
    """Make prediction using the loaded model"""
    try:
        calibrated_model = model_info['model']
        prediction_proba = calibrated_model.predict_proba(input_data)[0, 1]
        return prediction_proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_risk_level(probability, thresholds=[0.222, 0.64]):
    """Determine risk level based on probability and thresholds"""
    if probability < thresholds[0]:
        return "Low Risk", "risk-low", 0
    elif probability <= thresholds[1]:
        return "Medium Risk", "risk-medium", 1
    else:
        return "High Risk", "risk-high", 2

def create_risk_gauge(probability):
    """Create a simple risk gauge visualization"""
    # Create a simple gauge using HTML/CSS
    low_width = min(22.2, probability * 100)
    medium_width = min(64 - 22.2, max(0, probability * 100 - 22.2))
    high_width = max(0, probability * 100 - 64)
    
    gauge_html = f"""
    <div style="margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="color: #28a745;">Low Risk</span>
            <span style="color: #ffc107;">Medium Risk</span>
            <span style="color: #dc3545;">High Risk</span>
        </div>
        <div style="background: #f0f0f0; border-radius: 10px; height: 30px; overflow: hidden; position: relative;">
            <div style="background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%); 
                        width: 100%; height: 100%; opacity: 0.3;"></div>
            <div style="background: #007bff; height: 100%; width: {probability * 100}%; 
                        border-radius: 10px; position: absolute; top: 0; left: 0;"></div>
            <div style="position: absolute; top: 0; left: {probability * 100}%; 
                        height: 100%; width: 3px; background: #000;"></div>
        </div>
        <div style="text-align: center; margin-top: 10px; font-weight: bold; font-size: 18px;">
            Risk Score: {probability * 100:.1f}%
        </div>
    </div>
    """
    return gauge_html

def get_clinical_recommendations(risk_level):
    """Get clinical recommendations based on risk level"""
    recommendations = {
        "Low Risk": {
            "title": "🟢 Low Risk Clinical Recommendations",
            "color": "#28a745",
            "recommendations": [
                "📊 **Continue routine clinical monitoring**"
            ]
        },
        "Medium Risk": {
            "title": "🟡 Medium Risk Clinical Recommendations",
            "color": "#ffc107",
            "recommendations": [
                "📊 **Increase monitoring frequency**",
                "🩺 **Consider fibrin degradation products (FDP), antithrombin III等的监测**"
            ]
        },
        "High Risk": {
            "title": "🔴 High Risk Clinical Recommendations",
            "color": "#dc3545",
            "recommendations": [
                "🚨 **Immediate Action: Urgent consultation required**",
                "💉 **STAT repeat coagulation panel, CBC, fibrinogen, renal and liver function**",
                "📞 **Alert senior clinician, activate rapid response team if clinical deterioration**",
                "🩸 **Monitor for signs of microvascular thrombosis: digital ischemia, renal failure, altered mental status**"
            ],
            "actions": [
                "Monitor for Complications:",
                "• Acute kidney injury",
                "• ARDS",
                "• Hepatic failure",
                "• Cerebral ischemia"
            ]
        }
    }
    return recommendations.get(risk_level, {})

def create_parameter_display(values):
    """Create a visual display of parameter values with normal ranges"""
    normal_ranges = {
        "PT": (11.0, 13.5),
        "APTT": (25.0, 35.0),
        "D-Dimer": (0.0, 0.5),
        "MPV": (7.5, 11.5)
    }
    
    display_html = "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0;'>"
    
    for param, value in values.items():
        normal_min, normal_max = normal_ranges.get(param, (0, 0))
        is_abnormal = value < normal_min or value > normal_max
        
        color = "#dc3545" if is_abnormal else "#28a745"
        status = "⚠️ ABNORMAL" if is_abnormal else "✓ NORMAL"
        
        display_html += f"""
        <div style="flex: 1; min-width: 200px; background: white; padding: 15px; border-radius: 10px; 
                    border-left: 5px solid {color}; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <div style="font-weight: bold; font-size: 16px; margin-bottom: 8px;">{param}</div>
            <div style="font-size: 24px; font-weight: bold; color: {color}; margin-bottom: 5px;">
                {value:.1f}
            </div>
            <div style="font-size: 12px; color: #666;">
                Normal: {normal_min:.1f} - {normal_max:.1f}
            </div>
            <div style="font-size: 14px; font-weight: bold; color: {color}; margin-top: 5px;">
                {status}
            </div>
        </div>
        """
    
    display_html += "</div>"
    return display_html

def main():
    # Sidebar for additional information
    with st.sidebar:
        st.markdown("""
        <div class="card">
            <h3>📚 About PADM Model</h3>
            <p>The PADM model predicts Disseminated Intravascular Coagulation (DIC) risk using four key parameters:</p>
            <ul>
                <li><b>PT:</b> Prothrombin Time (s) - Measures extrinsic pathway</li>
                <li><b>APTT:</b> Activated Partial Thromboplastin Time (s) - Measures intrinsic pathway</li>
                <li><b>D-Dimer:</b> Fibrin degradation product (mg/L) - Marker of fibrinolysis</li>
                <li><b>MPV:</b> Mean Platelet Volume (fL) - Indicates platelet activation</li>
            </ul>
            <hr>
            <p><small>Validated in clinical studies (AUC: 0.904, Sensitivity: 79.2%, Specificity: 90.1%)</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>⚠️ Important Clinical Notes</h3>
            <p>This tool provides risk stratification but does NOT replace:</p>
            <ul>
                <li>Clinical judgment and examination</li>
                <li>Complete laboratory evaluation</li>
                <li>Consideration of underlying conditions</li>
                <li>Serial monitoring of parameters</li>
            </ul>
            <p>Always correlate with patient's clinical status.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Header
        st.markdown('<h1 class="main-header">⚕️ PADM DIC Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Predicting Disseminated Intravascular Coagulation Risk</p>', unsafe_allow_html=True)
        
        # Input section in card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">📝 Patient Parameters</h3>', unsafe_allow_html=True)
        
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            PT = st.number_input(
                "**Prothrombin Time (PT)**",
                min_value=0.0,
                max_value=100.0,
                value=12.0,
                step=0.1,
                help="Normal range: 11-13.5 seconds"
            )
            
            D_Dimer = st.number_input(
                "**D-Dimer**",
                min_value=0.0,
                max_value=50.0,
                value=0.5,
                step=0.1,
                help="Normal: <0.5 mg/L, Elevated: >0.5 mg/L"
            )
        
        with input_col2:
            APTT = st.number_input(
                "**Activated Partial Thromboplastin Time (APTT)**",
                min_value=0.0,
                max_value=200.0,
                value=30.0,
                step=0.1,
                help="Normal range: 25-35 seconds"
            )
            
            MPV = st.number_input(
                "**Mean Platelet Volume (MPV)**",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.1,
                help="Normal range: 7.5-11.5 fL"
            )
        
        # Prediction button
        if st.button("🔍 Calculate DIC Risk", use_container_width=True):
            st.session_state['prediction_made'] = True
            st.session_state['input_values'] = {
                'PT': PT,
                'APTT': APTT,
                'D-Dimer': D_Dimer,
                'MPV': MPV
            }
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Load model
        model_info = load_model()
        
        if model_info is None:
            st.error("Failed to load the prediction model. Please ensure PADM_model.pkl exists.")
            st.info("If you don't have a trained model, you can use the demo mode below.")
            
            # Demo mode
            if st.button("🔄 Use Demo Mode"):
                st.session_state['prediction_made'] = True
                st.session_state['input_values'] = {
                    'PT': 18.5,
                    'APTT': 55.0,
                    'D-Dimer': 3.2,
                    'MPV': 12.5
                }
            return
        
        # Display results if prediction was made
        if st.session_state.get('prediction_made', False):
            input_df = pd.DataFrame({
                'PT': [st.session_state['input_values']['PT']],
                'APTT': [st.session_state['input_values']['APTT']],
                'D-Dimer': [st.session_state['input_values']['D-Dimer']],
                'MPV': [st.session_state['input_values']['MPV']]
            })
            
            probability = predict_risk(model_info, input_df)
            
            if probability is not None:
                # Results card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3 class="sub-header">📊 Prediction Results</h3>', unsafe_allow_html=True)
                
                # Risk level and probability
                risk_level, risk_class, risk_idx = get_risk_level(probability)
                
                # Create metrics display
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="**DIC Probability**",
                        value=f"{probability:.1%}",
                        delta=None,
                        help="Probability of developing DIC"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    if risk_level == "Low Risk":
                        st.metric(label="**Risk Level**", value="🟢 LOW")
                    elif risk_level == "Medium Risk":
                        st.metric(label="**Risk Level**", value="🟡 MEDIUM")
                    else:
                        st.metric(label="**Risk Level**", value="🔴 HIGH")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Risk gauge visualization
                st.markdown("#### 📈 Risk Visualization")
                gauge_html = create_risk_gauge(probability)
                st.markdown(gauge_html, unsafe_allow_html=True)
                
                # Risk level display
                st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
                st.markdown(f"### {risk_level}")
                
                if risk_level == "Low Risk":
                    st.markdown("**Probability < 22.2%** - Routine monitoring recommended")
                elif risk_level == "Medium Risk":
                    st.markdown("**Probability 22.2% - 64.0%** - Increased vigilance required")
                else:
                    st.markdown("**Probability > 64.0%** - Urgent intervention needed")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Parameter visualization
                st.markdown("#### 📊 Parameter Analysis")
                param_display = create_parameter_display(st.session_state['input_values'])
                st.markdown(param_display, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Clinical recommendations
                st.markdown('<div class="card">', unsafe_allow_html=True)
                recommendations = get_clinical_recommendations(risk_level)
                
                if recommendations:
                    st.markdown(f"### {recommendations['title']}")
                    
                    st.markdown("#### 📋 Recommended Actions")
                    for rec in recommendations['recommendations']:
                        st.markdown(f"• {rec}")
                    
                    # 只有高风险有额外行动项
                    if risk_level == "High Risk" and 'actions' in recommendations:
                        st.markdown("#### ⚡ Immediate Considerations")
                        for action in recommendations['actions']:
                            st.markdown(f"{action}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # Placeholder before prediction
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">📊 Prediction Results</h3>', unsafe_allow_html=True)
            st.info("👈 Enter patient parameters and click 'Calculate DIC Risk' to see predictions")
            
            # Display sample risk gauge
            sample_gauge = create_risk_gauge(0)
            st.markdown(sample_gauge, unsafe_allow_html=True)
            
            st.markdown("""
            #### 🎯 Risk Categories
            - **🟢 Low Risk**: <22.2% probability
            - **🟡 Medium Risk**: 22.2%-64% probability  
            - **🔴 High Risk**: >64% probability
            """)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Sample recommendations placeholder
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📋 Clinical Recommendations")
            st.info("Risk-specific clinical guidance will appear here after calculation")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    col_footer1, col_footer2 = st.columns(2)
    
    with col_footer1:
        st.markdown("**⚠️ Disclaimer**")
        st.markdown("*Clinical decision support tool only*")
    
    with col_footer2:
        st.markdown("**📅 Version**")
        st.markdown(f"PADM v1.0 • {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    if 'prediction_made' not in st.session_state:
        st.session_state['prediction_made'] = False
    if 'input_values' not in st.session_state:
        st.session_state['input_values'] = {}
    
    main()