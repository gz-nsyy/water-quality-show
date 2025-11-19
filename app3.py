import warnings
import tempfile
import base64
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def setup_font():
    
    try:
        
        matplotlib.font_manager._rebuild()
        
        
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif'],
            'axes.unicode_minus': False,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
        return True
    except:
        
        return False


setup_font()


def main():
    st.set_page_config(
        page_title="Microbial Water Quality Prediction System",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🌊 Microbial Water Quality Prediction System")
    st.markdown("---")

    
    st.sidebar.header("🔧 Function Modules")

  
    if st.sidebar.button("📈 Temporal Analysis", use_container_width=True):
        st.session_state.current_function = "Temporal Analysis"
    
    if st.sidebar.button("🔗 Multimodal Analysis", use_container_width=True):
        st.session_state.current_function = "Multimodal Analysis"
    
    if st.sidebar.button("🤖 Machine Learning", use_container_width=True):
        st.session_state.current_function = "Machine Learning"
    
    if st.sidebar.button("🔬 Feature Importance", use_container_width=True):
        st.session_state.current_function = "Feature Importance"
    
    if st.sidebar.button("🔮 Time Series Forecast", use_container_width=True):
        st.session_state.current_function = "Time Series Forecast"
    
    if st.sidebar.button("📊 Risk Trend Analysis", use_container_width=True):
        st.session_state.current_function = "Risk Trend Analysis"

   
    if 'current_function' not in st.session_state:
        st.session_state.current_function = "Temporal Analysis"

    
    st.header(f"📋 {st.session_state.current_function} Demo")

   
    if st.session_state.current_function == "Temporal Analysis":
        show_temporal_analysis()
    
    elif st.session_state.current_function == "Multimodal Analysis":
        show_multimodal_analysis()
    
    elif st.session_state.current_function == "Machine Learning":
        show_machine_learning()
    
    elif st.session_state.current_function == "Feature Importance":
        show_feature_importance()
    
    elif st.session_state.current_function == "Time Series Forecast":
        show_time_series_forecast()
    
    elif st.session_state.current_function == "Risk Trend Analysis":
        show_risk_trend_analysis()

    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **System Features**
    - 📈 Temporal Dynamics Monitoring
    - 🔗 Multi-source Data Fusion  
    - 🤖 Intelligent Model Prediction
    - 🔬 Deep Feature Analysis
    - 🔮 Trend Prediction & Warning
    - 📊 Risk Assessment Management
    """)



def create_simple_figure():
    
    fig, ax = plt.subplots()
    
    ax.grid(True, alpha=0.3)
    return fig, ax

def show_temporal_analysis():
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔄 Microbial Community Dynamics")
        st.markdown("""
        **Core Functions:**
        - Species richness tracking
        - Community stability calculation
        - Key OTU trajectory analysis
        - Time trend visualization
        """)
        
       
        fig, ax = create_simple_figure()
        time_points = range(1, 13)
        richness = [50, 55, 52, 58, 60, 62, 65, 63, 68, 70, 72, 75]
        ax.plot(time_points, richness, 'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Time (Months)')
        ax.set_ylabel('Species Richness')
        ax.set_title('Temporal Changes in Species Richness')
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Dynamic Metrics")
        st.metric("Average Richness", "62.5", "+12.5%")
        st.metric("Stability Index", "0.85", "+0.05")
        st.metric("Trend", "Increasing", "Positive")

def show_multimodal_analysis():
    
    st.subheader("🌐 Multi-source Data Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Integration:**
        - Microbial community data
        - Physicochemical indicators
        - Meteorological data
        - Spatiotemporal analysis
        """)
        
        st.info("""
        **Supported Data Types:**
        - OTU abundance matrix
        - pH, DO, COD, etc.
        - Temperature, precipitation, humidity
        """)
    
    with col2:
        st.markdown("""
        **Analysis Capabilities:**
        - Cross-modal correlation
        - Feature interaction networks
        - Multi-dimensional association
        - Comprehensive metrics
        """)
        
        
        fig, ax = create_simple_figure()
        features = ['pH', 'DO', 'Temp', 'OTU1', 'OTU2', 'OTU3']
        corr_matrix = np.random.uniform(-0.8, 0.8, (6, 6))
        np.fill_diagonal(corr_matrix, 1)
        
        
        sns.heatmap(corr_matrix, cmap='RdBu_r', center=0,
                   xticklabels=features, yticklabels=features, ax=ax,
                   cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)

def show_machine_learning():
    
    st.subheader("🧠 Intelligent Prediction Models")
    
    tab1, tab2, tab3 = st.tabs(["Model Types", "Performance", "Confusion Matrix"])
    
    with tab1:
        st.markdown("""
        **Supported Algorithms:**
        - Logistic Regression (LR)
        - Support Vector Machine (SVM)
        - Random Forest (RF)
        - OneVsRest Multi-class
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Select Model", ['LR', 'SVML', 'SVMRBF', 'RF'], index=0)
            st.checkbox("Apply SMOTE", value=True)
        
        with col2:
            st.slider("CV Folds", 2, 10, 5)
            st.slider("Test Size", 0.1, 0.5, 0.3)
    
    with tab2:
        # 性能表格
        performance_data = {
            'Model': ['LR', 'SVM Linear', 'SVM RBF', 'Random Forest'],
            'Accuracy': [0.85, 0.88, 0.92, 0.94],
            'AUC': [0.89, 0.91, 0.95, 0.96],
            'F1 Score': [0.84, 0.87, 0.91, 0.93]
        }
        st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
    
    with tab3:
        
        fig, ax = create_simple_figure()
        classes = ['Clean', 'Light', 'Heavy']
        cm = np.array([[25, 2, 1], [1, 28, 3], [0, 1, 29]])
        
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes, ax=ax,
                   cbar_kws={'shrink': 0.8})
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

def show_feature_importance():
   
    st.subheader("🔍 Deep Feature Analysis")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        
        fig, ax = create_simple_figure()
        features = [f'OTU_{i}' for i in range(1, 11)]
        importance = np.random.uniform(0.05, 0.2, 10)
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, color='steelblue', alpha=0.8, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Important Features')
        ax.invert_yaxis()
        st.pyplot(fig)
    
    with col2:
        
        **Analysis Methods:**
        - SHAP value analysis
        - LR coefficient weights
        - Log2 ratio calculation
        - Statistical significance
        - Multiple testing correction
        """)
        
        st.success("""
        **Output Results:**
        - Feature importance ranking
        - Interaction networks
        - Biomarker identification
        - Interpretability analysis
        """)

def show_time_series_forecast():
    
    st.subheader("📈 Future Trend Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
       
        fig, ax = create_simple_figure()
        
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        history_values = np.random.normal(0.3, 0.05, 12) + np.linspace(0, 0.1, 12)
        
      
        future_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        forecast_values = np.random.normal(0.4, 0.03, 6)
        
       
        all_months = months + future_months
        all_values = list(history_values) + list(forecast_values)
        
        ax.plot(range(len(months)), history_values, 'b-o', label='Historical', linewidth=2, markersize=4)
        ax.plot(range(len(months), len(all_months)), forecast_values, 'r--o', label='Forecast', linewidth=2, markersize=4)
        ax.axhline(y=0.35, color='red', linestyle=':', alpha=0.7, label='Risk Threshold')
        
        
        ax.set_xticks(range(len(all_months)))
        ax.set_xticklabels(all_months, rotation=45)
        ax.set_xlabel('Time')
        ax.set_ylabel('Risk Ratio')
        ax.set_title('Risk Trend Prediction')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        **Prediction Methods:**
        - ARIMA model
        - Moving average
        - Exponential smoothing
        - Prophet algorithm
        """)
        
        st.warning("""
        **Warning Information:**
        - High-risk period detection
        - Trend change alerts
        - Anomaly detection
        """)
        
        st.metric("Prediction Accuracy", "89.2%", "+2.1%")
        st.metric("Early Warning", "15 days", "+3 days")

def show_risk_trend_analysis():
    
    st.subheader("⚠️ Risk Assessment & Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Risk Level", "Medium", "Stable")
        st.metric("Risk Trend", "Increasing", "+0.05")
    
    with col2:
        st.metric("Warning Days", "12 days", "+2 days")
        st.metric("Confidence", "92%", "+3%")
    
    with col3:
        st.metric("Key Indicator", "OTU_157", "High Risk")
        st.metric("Impact Level", "High", "↑")
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = create_simple_figure()
        risk_levels = ['Low', 'Medium', 'High']
        risk_counts = [25, 15, 8]
        colors = ['green', 'orange', 'red']
        bars = ax.bar(risk_levels, risk_counts, color=colors, alpha=0.8)
        ax.set_title('Risk Level Distribution')
        ax.set_ylabel('Sample Count')
        # 在柱子上添加数值
        for bar, count in zip(bars, risk_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom')
        st.pyplot(fig)
    
    with col2:
        fig, ax = create_simple_figure()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        risk_scores = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        ax.plot(months, risk_scores, 'r-o', linewidth=2, markersize=4)
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Threshold')
        ax.fill_between(months, risk_scores, 0.3, where=np.array(risk_scores) > 0.3, 
                       color='red', alpha=0.1, label='High Risk') color='red', alpha=0.1, label='高风险')
        ax.set_title('Monthly Risk Trend')
        ax.set_ylabel('Risk Score')
        ax.legend()
        st.pyplot(fig)
    
    st.info("""
    **Risk Management Functions:**
        - Real-time risk monitoring
- 实时风险监控
        - Trend prediction & warnings
- 关键因素识别
        - Key factor identification
        - Prevention recommendations
    """)

if __name__ == "__main__":

    main()

