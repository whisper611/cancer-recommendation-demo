import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# 设置页面
st.set_page_config(
    page_title="Precision Oncology Recommendation System",
    layout="wide",
    page_icon="⚕️"
)
st.title("⚕️ Precision Oncology Content Recommendation System")
st.markdown("""
Personalized article recommendations for oncology specialists based on professional profile and interests.
""")

# 设置样式
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        .st-eb {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
        }
        .css-1d391kg {padding-top: 0rem;}
    </style>
""", unsafe_allow_html=True)

# 肿瘤类型列表
cancer_types = ["Lung Cancer", "Breast Cancer", "Colorectal Cancer", "Gastric Cancer", 
                "Liver Cancer", "Pancreatic Cancer", "Lymphoma", "Leukemia", 
                "Prostate Cancer", "Ovarian Cancer", "Cervical Cancer", "Bladder Cancer", 
                "Renal Cancer", "Thyroid Cancer", "Nasopharyngeal Cancer"]

# 科室列表
departments = ["Medical Oncology", "Surgical Oncology", "Radiation Oncology", 
               "Interventional Oncology", "Pathology", "Radiology", "Nuclear Medicine"]

# ======================
# 1. 模拟数据
# ======================
@st.cache_data
def generate_doctors(num=100):
    np.random.seed(42)
    data = {
        "doctor_id": range(1, num+1),
        "name": [f"Doctor {str(i).zfill(3)}" for i in range(1, num+1)],
        "level": np.random.choice(["Resident", "Attending", "Associate Professor", "Professor"], num, p=[0.2, 0.4, 0.3, 0.1]),
        "hospital_level": np.random.choice(["Community Hospital", "Secondary Hospital", "Tertiary B", "Tertiary A"], num, p=[0.1, 0.3, 0.3, 0.3]),
        "department": np.random.choice(departments, num),
        "specialty": [np.random.choice(cancer_types, np.random.randint(1, 4), replace=False).tolist() for _ in range(num)],
        "years_exp": np.random.randint(1, 35, num)
    }
    return pd.DataFrame(data)

@st.cache_data
def generate_articles(num=200):
    np.random.seed(42)
    
    # 文章类型
    article_types = [
        "Clinical Research", "Guideline Update", "Case Discussion", "Review", 
        "Clinical Trial", "Technical Advance", "Expert Consensus", 
        "Research Brief", "New Treatment Strategy", "Drug Evaluation", 
        "Diagnostic Breakthrough", "Prognostic Model", "Multicenter Study"
    ]
    
    # 标题元素
    title_prefixes = [
        "Advances in Oncology: ", "New Research: ", "Breakthrough: ", 
        "Clinical Practice: ", "Expert Perspective: ", "Technology Update: ",
        "Treatment Options: ", "Diagnostic Methods: ", "Prognostic Assessment: ",
        "Case Study: ", "Guideline Review: ", "Comprehensive Analysis: "
    ]
    
    # 主题关键词
    themes = [
        "targeted therapy", "immunotherapy", "precision radiotherapy", 
        "early screening", "molecular subtyping", "drug resistance mechanisms",
        "biomarkers", "genetic testing", "liquid biopsy", "tumor microenvironment",
        "combination therapy", "neoadjuvant therapy", "adjuvant therapy", 
        "palliative care", "quality of life"
    ]
    
    data = {
        "article_id": range(1, num+1),
        "title": [],
        "type": np.random.choice(article_types, num),
        "cancer_type": [np.random.choice(cancer_types, np.random.randint(1, 3), replace=False).tolist() for _ in range(num)],
        "department": [np.random.choice(departments, np.random.randint(1, 3), replace=False).tolist() for _ in range(num)],
        "authority": np.random.choice(["Top International Journal", "National Core Journal", "Society Guideline", "Conference Abstract"], num, p=[0.1, 0.4, 0.3, 0.2]),
        "impact_factor": np.random.uniform(0, 30, num).round(1),
        "pub_date": pd.date_range(start="2023-01-01", end="2024-05-01", periods=num),
        "keywords": [[np.random.choice(themes, np.random.randint(3, 6), replace=False).tolist()] for _ in range(num)]
    }
    
    # 生成标题
    for i in range(num):
        cancer = np.random.choice(cancer_types)
        theme = np.random.choice(themes)
        
        title_type = np.random.choice(["prefix", "simple", "question", "numbered"], p=[0.4, 0.3, 0.2, 0.1])
        
        if title_type == "prefix":
            prefix = np.random.choice(title_prefixes)
            title = f"{prefix}{theme} in {cancer}"
        elif title_type == "simple":
            title = f"Recent Advances in {cancer} {theme}"
        elif title_type == "question":
            questions = [
                f"{theme} in {cancer}: Survival Benefit?",
                f"Optimizing {theme} Strategies for {cancer}",
                f"Clinical Value of {theme} in {cancer}"
            ]
            title = np.random.choice(questions)
        else:  # numbered
            title = f"Study Report {i+1}: {theme} in {cancer}"
        
        data["title"].append(title)
    
    return pd.DataFrame(data)

# 生成数据
doctors = generate_doctors(100)
articles = generate_articles(200)

# ======================
# 2. 特征工程
# ======================
def encode_features(doctors_df, articles_df):
    # 医生特征编码
    doctor_features = pd.DataFrame()
    
    # 职级编码
    level_map = {"Resident": 1, "Attending": 2, "Associate Professor": 3, "Professor": 4}
    doctor_features["level"] = doctors_df["level"].map(level_map).fillna(0)
    
    # 医院等级编码
    hospital_map = {"Community Hospital": 1, "Secondary Hospital": 2, "Tertiary B": 3, "Tertiary A": 4}
    doctor_features["hospital_level"] = doctors_df["hospital_level"].map(hospital_map).fillna(0)
    
    # 获取所有科室和瘤肿的统一列表
    all_departments = list(set(doctors_df['department'].unique().tolist() + 
                             [dept for sublist in articles_df['department'] for dept in sublist]))
    
    all_cancers = list(set([c for sublist in doctors_df['specialty'] for c in sublist] + 
                           [c for sublist in articles_df['cancer_type'] for c in sublist]))
    
    # 科室OneHot编码 - 使用统一的维度
    dept_encoder = OneHotEncoder(categories=[all_departments], sparse_output=False, handle_unknown='ignore')
    # 先拟合空数据以创建categories_属性
    dept_encoder.fit(doctors_df[["department"]])
    dept_encoded = dept_encoder.transform(doctors_df[["department"]])
    dept_cols = [f"dept_{dept}" for dept in dept_encoder.categories_[0]]
    doctor_features[dept_cols] = dept_encoded
    
    # 擅长瘤肿编码 - 使用统一的维度
    cancer_encoder = OneHotEncoder(categories=[all_cancers], sparse_output=False, handle_unknown='ignore')
    # 先拟合空数据以创建categories_属性
    cancer_encoder.fit(doctors_df[["department"]])  # 任意数据，只为初始化
    
    cancer_features = []
    for specialties in doctors_df["specialty"]:
        vec = np.zeros(len(cancer_encoder.categories_[0]))
        for spec in specialties:
            if spec in cancer_encoder.categories_[0]:
                idx = np.where(cancer_encoder.categories_[0] == spec)[0][0]
                vec[idx] = 1
        cancer_features.append(vec)
    
    cancer_cols = [f"cancer_{c}" for c in cancer_encoder.categories_[0]]
    doctor_features[cancer_cols] = cancer_features
    
    # 工作经验标准化
    max_exp = doctors_df["years_exp"].max()
    if max_exp > 0:
        doctor_features["years_exp"] = doctors_df["years_exp"] / max_exp
    else:
        doctor_features["years_exp"] = 0
    
    # 文章特征编码
    article_features = pd.DataFrame()
    
    # 文章类型OneHot编码
    all_article_types = articles_df['type'].unique().tolist()
    type_encoder = OneHotEncoder(categories=[all_article_types], sparse_output=False, handle_unknown='ignore')
    type_encoder.fit(articles_df[["type"]])  # 拟合以创建categories_属性
    type_encoded = type_encoder.transform(articles_df[["type"]])
    type_cols = [f"type_{t}" for t in type_encoder.categories_[0]]
    article_features[type_cols] = type_encoded
    
    # 文章瘤肿类型编码 - 使用统一的维度
    cancer_features_art = []
    for cancers in articles_df["cancer_type"]:
        vec = np.zeros(len(cancer_encoder.categories_[0]))
        for cancer in cancers:
            if cancer in cancer_encoder.categories_[0]:
                idx = np.where(cancer_encoder.categories_[0] == cancer)[0][0]
                vec[idx] = 1
        cancer_features_art.append(vec)
    
    article_features[cancer_cols] = cancer_features_art
    
    # 文章科室编码 - 使用统一的维度
    dept_features_art = []
    for depts in articles_df["department"]:
        vec = np.zeros(len(dept_encoder.categories_[0]))
        for dept in depts:
            if dept in dept_encoder.categories_[0]:
                idx = np.where(dept_encoder.categories_[0] == dept)[0][0]
                vec[idx] = 1
        dept_features_art.append(vec)
    
    article_features[dept_cols] = dept_features_art
    
    # 权威性编码
    authority_map = {"Conference Abstract": 1, "National Core Journal": 2, "Society Guideline": 3, "Top International Journal": 4}
    article_features["authority"] = articles_df["authority"].map(authority_map).fillna(0)
    
    # 影响因子标准化
    max_impact = articles_df["impact_factor"].max()
    if max_impact > 0:
        article_features["impact_factor"] = articles_df["impact_factor"] / max_impact
    else:
        article_features["impact_factor"] = 0
    
    # 时间衰减因子
    max_date = articles_df["pub_date"].max()
    if not pd.isnull(max_date):
        article_features["recency"] = (max_date - articles_df["pub_date"]).dt.days.apply(
            lambda x: np.exp(-x/180) if not pd.isnull(x) else 0
        )
    else:
        article_features["recency"] = 0
    
    # 确保医生和文章特征维度一致
    # 添加缺失的列并填充0
    for col in doctor_features.columns:
        if col not in article_features:
            article_features[col] = 0
    
    for col in article_features.columns:
        if col not in doctor_features:
            doctor_features[col] = 0
    
    # 确保列顺序一致
    doctor_features = doctor_features[article_features.columns]
    
    # 处理空值问题 - 确保没有 NaN
    doctor_features = doctor_features.fillna(0)
    article_features = article_features.fillna(0)
    
    return doctor_features, article_features, dept_cols, cancer_cols, type_cols

# 执行特征编码
doctor_features, article_features, dept_cols, cancer_cols, type_cols = encode_features(doctors, articles)

# ======================
# 3. 推荐算法
# ======================
def recommend_articles(doctor_id, top_n=5):
    """为指定医生推荐文章"""
    doctor_idx = doctor_id - 1
    
    # 获取医生向量
    doctor_vec = doctor_features.iloc[doctor_idx].values.reshape(1, -1)
    
    # 验证向量没有 NaN
    if np.isnan(doctor_vec).any():
        doctor_vec = np.nan_to_num(doctor_vec)
    
    # 获取文章特征矩阵
    article_vecs = article_features.values
    
    # 验证矩阵没有 NaN
    if np.isnan(article_vecs).any():
        article_vecs = np.nan_to_num(article_vecs)
    
    # 计算用户向量与所有文章向量的余弦相似度
    similarity_scores = cosine_similarity(doctor_vec, article_vecs)[0]
    
    # 获取推荐排序
    ranked_articles = np.argsort(similarity_scores)[::-1][:top_n]
    
    # 返回推荐结果和相似度分数
    recommended = articles.iloc[ranked_articles].copy()
    recommended["match_score"] = similarity_scores[ranked_articles].round(3)
    
    return recommended

# ======================
# 4. Streamlit交互界面
# ======================
# 侧边栏 - 用户选择
st.sidebar.header("👨‍⚕️ Doctor Profile Settings")
selected_doctor = st.sidebar.selectbox("Select Doctor Profile", doctors["name"])

# 获取选定医生的信息
doctor_info = doctors[doctors["name"] == selected_doctor].iloc[0]

# 显示医生信息
st.sidebar.subheader("Doctor Profile Details")
st.sidebar.markdown(f"**Level:** {doctor_info['level']}")
st.sidebar.markdown(f"**Hospital Level:** {doctor_info['hospital_level']}")
st.sidebar.markdown(f"**Department:** {doctor_info['department']}")
st.sidebar.markdown(f"**Specialty Cancers:** {', '.join(doctor_info['specialty']}")
st.sidebar.markdown(f"**Years of Experience:** {doctor_info['years_exp']} years")

# 推荐设置
st.sidebar.subheader("Recommendation Settings")
top_n = st.sidebar.slider("Number of Recommendations", 3, 10, 5)
weight_specialty = st.sidebar.slider("Specialty Cancer Weight", 0.0, 1.0, 0.6)
weight_department = st.sidebar.slider("Department Weight", 0.0, 1.0, 0.3)
weight_level = st.sidebar.slider("Seniority Weight", 0.0, 1.0, 0.1)

# 调整权重
doctor_features_weighted = doctor_features.copy()
doctor_features_weighted[cancer_cols] *= weight_specialty
doctor_features_weighted[dept_cols] *= weight_department
doctor_features_weighted["level"] *= weight_level

# 主界面
st.header(f"📰 Recommended Oncology Articles for {selected_doctor}")

# 添加进度条
progress_bar = st.progress(0)
status_text = st.empty()

# 模拟计算过程
for i in range(100):
    progress_bar.progress(i + 1)
    status_text.text(f"Finding best matches... {i+1}%")
    time.sleep(0.01)

# 获取推荐结果
recommended_articles = recommend_articles(doctor_info["doctor_id"], top_n)

# 显示推荐结果
for idx, row in recommended_articles.iterrows():
    with st.expander(f"**[{row['article_id']}] {row['title']}** (Match Score: {row['match_score']:.2f})", expanded=idx==0):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # 文章类型图标
            type_icon = {
                "Clinical Research": "🔬",
                "Guideline Update": "📜",
                "Case Discussion": "📋",
                "Review": "📚",
                "Clinical Trial": "💊",
                "Technical Advance": "⚙️",
                "Expert Consensus": "👥",
                "Research Brief": "📰",
                "New Treatment Strategy": "💡",
                "Drug Evaluation": "💊",
                "Diagnostic Breakthrough": "🔍",
                "Prognostic Model": "📊",
                "Multicenter Study": "🌐"
            }.get(row["type"], "📄")
            
            st.markdown(f"**Type:** {type_icon} {row['type']}")
            st.markdown(f"**Related Cancers:** {', '.join(row['cancer_type']}")
            st.markdown(f"**Related Departments:** {', '.join(row['department']}")
            st.markdown(f"**Source:** {row['authority']}")
            st.markdown(f"**Impact Factor:** {row['impact_factor']}")
            st.markdown(f"**Publication Date:** {row['pub_date'].strftime('%Y-%m-%d')}")
        
        with col2:
            # 生成匹配理由
            match_reasons = []
            
            # 科室匹配
            if any(dept in row["department"] for dept in [doctor_info["department"]]):
                match_reasons.append(f"Related to your department ({doctor_info['department']})")
            
            # 瘤肿匹配
            common_cancers = set(doctor_info["specialty"]) & set(row["cancer_type"])
            if common_cancers:
                match_reasons.append(f"Matches your specialty: {', '.join(common_cancers)}")
            
            # 职级匹配
            if doctor_info["level"] in ["Associate Professor", "Professor"] and row["type"] in ["Guideline Update", "Expert Consensus", "Review"]:
                match_reasons.append("Suited for senior specialists")
            elif doctor_info["level"] in ["Resident", "Attending"] and row["type"] in ["Case Discussion", "Clinical Research", "New Treatment Strategy"]:
                match_reasons.append("Relevant for clinical learning")
            
            # 影响因子
            if row["impact_factor"] > 15:
                match_reasons.append("High-impact publication")
            
            # 显示匹配理由
            if match_reasons:
                st.info("**Match Reasons:** " + "; ".join(match_reasons))
            else:
                st.info("**Match Reasons:** Comprehensive feature matching")
            
            # 显示关键词
            keywords = row["keywords"][0] if isinstance(row["keywords"], list) else row["keywords"]
            st.markdown("**Keywords:** " + ", ".join(keywords))
            
            # 生成摘要
            cancer = np.random.choice(row["cancer_type"])
            keywords_str = ", ".join(keywords[:3])
            st.markdown(f"""
            **Abstract:**  
            This study explores the application of {keywords_str} in {cancer} treatment. Analysis of {np.random.randint(50, 500)} patients demonstrated significant improvements in {np.random.choice(["overall survival", "quality of life", "recurrence risk reduction", "treatment safety"])}. 
            These findings provide new evidence-based insights for clinical practice in {cancer}.
            """)

# 添加分隔线
st.markdown("---")

# 推荐分析图表
st.subheader("Recommendation Analysis")
col1, col2 = st.columns(2)

with col1:
    # 文章类型分布
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # 获取类型计数并排序
        type_counts = recommended_articles["type"].value_counts()
        sorted_types = type_counts.index.tolist()
        
        # 创建条形图
        ax.barh(sorted_types, type_counts.values, color='#1f77b4')
        
        # 设置英文标签
        ax.set_title("Distribution of Recommended Article Types")
        ax.set_xlabel("Count")
        ax.set_ylabel("Article Type")
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating type distribution: {str(e)}")
        st.write("Distribution of Recommended Article Types:")
        st.dataframe(type_counts)

with col2:
    # 匹配分数分布
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # 创建直方图
        sns.histplot(recommended_articles["match_score"], bins=6, kde=True, ax=ax, color='#ff7f0e')
        
        # 设置英文标签
        ax.set_title("Distribution of Match Scores")
        ax.set_xlabel("Match Score")
        ax.set_ylabel("Number of Articles")
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating score distribution: {str(e)}")
        st.write("Match Score Statistics:")
        st.write(recommended_articles["match_score"].describe())

# 系统说明
st.markdown("---")
st.subheader("System Description")
st.markdown("""
This recommendation system personalizes content based on:
1. **Professional Level**: Different content for residents vs. professors
2. **Hospital Tier**: Resource availability considerations
3. **Department**: Specialty-specific relevance
4. **Cancer Specialties**: Focus on areas of expertise
5. **Experience**: Advanced research for experienced specialists

The algorithm uses feature vectorization and cosine similarity with adjustable weights.
""")

# 隐藏进度条
progress_bar.empty()
status_text.empty()
