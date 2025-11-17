# =========================
# 导入必要的库
# =========================
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

# =========================
# 修复字体配置
# =========================
def setup_font():
    """配置中文字体支持"""
    try:
        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False

# 初始化字体
setup_font()

# =========================
# 功能展示主程序
# =========================

def main():
    st.set_page_config(
        page_title="微生物水质预测分析系统",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🌊 微生物指标预测水质指标模型开发系统")
    st.markdown("---")

    # 侧边栏功能选择
    st.sidebar.header("🔧 功能模块")

    # 功能选择按钮
    if st.sidebar.button("📈 时序分析", use_container_width=True):
        st.session_state.current_function = "时序分析"
    
    if st.sidebar.button("🔗 多模态分析", use_container_width=True):
        st.session_state.current_function = "多模态分析"
    
    if st.sidebar.button("🤖 机器学习建模", use_container_width=True):
        st.session_state.current_function = "机器学习建模"
    
    if st.sidebar.button("🔬 特征重要性分析", use_container_width=True):
        st.session_state.current_function = "特征重要性分析"
    
    if st.sidebar.button("🔮 时间序列预测", use_container_width=True):
        st.session_state.current_function = "时间序列预测"
    
    if st.sidebar.button("📊 风险趋势分析", use_container_width=True):
        st.session_state.current_function = "风险趋势分析"

    # 初始化会话状态
    if 'current_function' not in st.session_state:
        st.session_state.current_function = "时序分析"

    # 显示当前功能说明
    st.header(f"📋 {st.session_state.current_function} 功能展示")

    # 各功能模块的展示内容
    if st.session_state.current_function == "时序分析":
        show_temporal_analysis()
    
    elif st.session_state.current_function == "多模态分析":
        show_multimodal_analysis()
    
    elif st.session_state.current_function == "机器学习建模":
        show_machine_learning()
    
    elif st.session_state.current_function == "特征重要性分析":
        show_feature_importance()
    
    elif st.session_state.current_function == "时间序列预测":
        show_time_series_forecast()
    
    elif st.session_state.current_function == "风险趋势分析":
        show_risk_trend_analysis()

    # 系统信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **系统功能概览**
    - 📈 时序动态监测
    - 🔗 多源数据融合  
    - 🤖 智能模型预测
    - 🔬 深度特征解析
    - 🔮 趋势预测预警
    - 📊 风险评估管理
    """)

# =========================
# 各功能展示函数
# =========================

def show_temporal_analysis():
    """时序分析功能展示"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔄 微生物群落时序动态")
        st.markdown("""
        **核心功能：**
        - 物种丰富度变化追踪
        - 群落稳定性指数计算
        - 关键OTU轨迹分析
        - 时间趋势可视化
        """)
        
        # 模拟图表展示
        fig, ax = plt.subplots(figsize=(10, 6))
        time_points = range(1, 13)  # 12个月
        richness = [50, 55, 52, 58, 60, 62, 65, 63, 68, 70, 72, 75]  # 12个数据点
        ax.plot(time_points, richness, 'b-o', linewidth=2)
        ax.set_xlabel('时间 (月)', fontsize=12)
        ax.set_ylabel('物种丰富度', fontsize=12)
        ax.set_title('物种丰富度时序变化', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 动态指标")
        st.metric("平均丰富度", "62.5", "↑ 12.5%")
        st.metric("稳定性指数", "0.85", "↑ 0.05")
        st.metric("变化趋势", "上升", "积极")

def show_multimodal_analysis():
    """多模态分析功能展示"""
    st.subheader("🌐 多源数据融合分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **数据整合：**
        - 微生物群落数据
        - 理化指标数据
        - 气象环境数据
        - 时空关联分析
        """)
        
        st.info("""
        **支持的数据类型：**
        - OTU丰度矩阵
        - pH、DO、COD等理化指标
        - 温度、降水、湿度等气象数据
        """)
    
    with col2:
        st.markdown("""
        **分析能力：**
        - 跨模态相关性分析
        - 特征交互网络
        - 多维度关联挖掘
        - 综合指标计算
        """)
        
        # 模拟相关性矩阵
        fig, ax = plt.subplots(figsize=(8, 6))
        features = ['pH', 'DO', '温度', 'OTU1', 'OTU2', 'OTU3']
        corr_matrix = np.random.uniform(-0.8, 0.8, (6, 6))
        np.fill_diagonal(corr_matrix, 1)
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=features, yticklabels=features, ax=ax)
        ax.set_title('多模态特征相关性', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig)

def show_machine_learning():
    """机器学习建模功能展示"""
    st.subheader("🧠 智能预测模型")
    
    tab1, tab2, tab3 = st.tabs(["模型类型", "性能指标", "混淆矩阵"])
    
    with tab1:
        st.markdown("""
        **支持的算法：**
        - 📊 逻辑回归 (LR)
        - 🔍 支持向量机 (SVM)
        - 🌳 随机森林 (RF)
        - 🔄 OneVsRest多分类
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("选择模型", ['LR', 'SVML', 'SVMRBF', 'RF'], index=0)
            st.checkbox("使用SMOTE过采样", value=True)
        
        with col2:
            st.slider("交叉验证折数", 2, 10, 5)
            st.slider("测试集比例", 0.1, 0.5, 0.3)
    
    with tab2:
        # 模拟性能表格
        performance_data = {
            '模型': ['LR', 'SVM线性', 'SVM径向基', '随机森林'],
            '准确率': [0.85, 0.88, 0.92, 0.94],
            'AUC': [0.89, 0.91, 0.95, 0.96],
            'F1分数': [0.84, 0.87, 0.91, 0.93]
        }
        st.dataframe(pd.DataFrame(performance_data))
    
    with tab3:
        # 模拟混淆矩阵
        fig, ax = plt.subplots(figsize=(6, 5))
        classes = ['清洁', '轻度污染', '重度污染']
        cm = np.array([[25, 2, 1], [1, 28, 3], [0, 1, 29]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title('混淆矩阵示例', fontsize=14, fontweight='bold')
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        st.pyplot(fig)

def show_feature_importance():
    """特征重要性分析功能展示"""
    st.subheader("🔍 深度特征解析")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # 模拟特征重要性图
        fig, ax = plt.subplots(figsize=(10, 8))
        features = [f'OTU_{i}' for i in range(1, 11)]
        importance = np.random.uniform(0.05, 0.2, 10)
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance, color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('特征重要性', fontsize=12)
        ax.set_title('Top 10 重要OTU特征', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        **分析方法：**
        - SHAP值分析
        - LR系数权重
        - Log2比值计算
        - 统计显著性检验
        - 多重比较校正
        """)
        
        st.success("""
        **输出结果：**
        - 特征重要性排名
        - 交互作用网络
        - 生物标志物识别
        - 可解释性分析
        """)

def show_time_series_forecast():
    """时间序列预测功能展示"""
    st.subheader("📈 未来趋势预测")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 模拟预测图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 修复：确保数据维度一致
        # 历史数据 - 12个月
        history_dates = pd.date_range('2023-01-01', periods=12, freq='M')
        history_values = np.random.normal(0.3, 0.05, 12) + np.linspace(0, 0.1, 12)
        
        # 预测数据 - 6个月
        forecast_dates = pd.date_range('2024-01-01', periods=6, freq='M')
        forecast_values = np.random.normal(0.4, 0.03, 6)
        
        # 验证数据维度
        st.write(f"历史数据维度: {len(history_dates)} 个时间点, {len(history_values)} 个值")
        st.write(f"预测数据维度: {len(forecast_dates)} 个时间点, {len(forecast_values)} 个值")
        
        ax.plot(history_dates, history_values, 'b-o', label='历史数据', linewidth=2)
        ax.plot(forecast_dates, forecast_values, 'r--o', label='ARIMA预测', linewidth=2)
        ax.axhline(y=0.35, color='red', linestyle=':', alpha=0.7, label='风险阈值')
        
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('污染风险比例', fontsize=12)
        ax.set_title('水质风险趋势预测', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        **预测方法：**
        - ARIMA模型
        - 移动平均法
        - 指数平滑
        - Prophet算法
        """)
        
        st.warning("""
        **预警信息：**
        - 高风险时段检测
        - 趋势变化预警
        - 异常波动提醒
        """)
        
        st.metric("预测准确率", "89.2%", "↑ 2.1%")
        st.metric("预警提前量", "15天", "↑ 3天")

def show_risk_trend_analysis():
    """风险趋势分析功能展示"""
    st.subheader("⚠️ 风险评估与管理")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("当前风险等级", "中等", "稳定")
        st.metric("风险趋势", "上升", "+0.05")
    
    with col2:
        st.metric("预警天数", "12天", "↑ 2天")
        st.metric("置信度", "92%", "↑ 3%")
    
    with col3:
        st.metric("关键指标", "OTU_157", "高风险")
        st.metric("影响程度", "高", "↑")
    
    # 风险分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 风险等级分布
    risk_levels = ['低风险', '中风险', '高风险']
    risk_counts = [25, 15, 8]
    colors = ['green', 'orange', 'red']
    ax1.bar(risk_levels, risk_counts, color=colors, alpha=0.8)
    ax1.set_title('风险等级分布', fontsize=14, fontweight='bold')
    ax1.set_ylabel('样本数量', fontsize=12)
    
    # 时间风险趋势 - 修复维度问题
    months = ['1月', '2月', '3月', '4月', '5月', '6月']  # 6个月
    risk_scores = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]  # 6个值
    ax2.plot(months, risk_scores, 'r-o', linewidth=2)
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='阈值')
    ax2.fill_between(months, risk_scores, 0.3, where=np.array(risk_scores) > 0.3, 
                    color='red', alpha=0.3, label='高风险区域')
    ax2.set_title('月度风险趋势', fontsize=14, fontweight='bold')
    ax2.set_ylabel('风险评分', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.info("""
    **风险管理功能：**
    - 实时风险监测
    - 趋势预测预警
    - 关键因子识别
    - 防控建议生成
    """)

if __name__ == "__main__":
    main()