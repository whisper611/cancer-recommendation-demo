import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import time


# 设置支持中文的字体
try:
    # Windows 系统
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        # 尝试其他常见中文字体
        for font in ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']:
            if font in [f.name for f in fm.fontManager.ttflist]:
                plt.rcParams['font.family'] = font
                break
    
    # 确保负号正常显示
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果找不到中文字体，使用支持中文的替代方案
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用 DejaVu Sans 作为备选
    st.warning("未找到中文字体，图表中的中文可能显示异常")

# 设置页面
st.set_page_config(page_title="肿瘤资讯精准推荐系统", layout="wide", page_icon="⚕️")
st.title("⚕️ 肿瘤医生专属资讯推荐系统")
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
    </style>
""", unsafe_allow_html=True)

# ======================
# 1. 模拟数据
# ======================
# 肿瘤类型列表
cancer_types = ["肺癌", "乳腺癌", "结直肠癌", "胃癌", "肝癌", "胰腺癌", "淋巴瘤", "白血病", 
                "前列腺癌", "卵巢癌", "宫颈癌", "膀胱癌", "肾癌", "甲状腺癌", "鼻咽癌"]

# 科室列表
departments = ["肿瘤内科", "肿瘤外科", "放射治疗科", "肿瘤介入科", "病理科", "影像科", "核医学科"]

# 创建模拟医生数据
def generate_doctors(num=100):
    np.random.seed(42)
    data = {
        "doctor_id": range(1, num+1),
        "name": [f"医生{str(i).zfill(3)}" for i in range(1, num+1)],
        "level": np.random.choice(["住院医师", "主治医师", "副主任医师", "主任医师"], num, p=[0.2, 0.4, 0.3, 0.1]),
        "hospital_level": np.random.choice(["社区医院", "二级医院", "三乙医院", "三甲医院"], num, p=[0.1, 0.3, 0.3, 0.3]),
        "department": np.random.choice(departments, num),
        "specialty": [np.random.choice(cancer_types, np.random.randint(1, 4), replace=False).tolist() for _ in range(num)],
        "years_exp": np.random.randint(1, 35, num)
    }
    return pd.DataFrame(data)

# 创建模拟文章数据
# 创建更丰富的模拟文章数据
def generate_articles(num=200):
    np.random.seed(42)
    
    # 更丰富的文章类型
    article_types = ["临床研究", "指南更新", "病例讨论", "综述", "临床试验", "技术进展", "专家共识", 
                    "研究简报", "治疗新策略", "药物评价", "诊断突破", "预后模型", "多中心研究"]
    
    # 更丰富的标题前缀
    title_prefixes = [
        "肿瘤领域重要进展：", "最新研究：", "突破性发现：", "临床实践：", "专家视角：", 
        "技术前沿：", "治疗新选择：", "诊断新方法：", "预后评估：", "病例分享：", 
        "指南解读：", "综述：", "多中心研究：", "真实世界研究：", "分子机制探索："
    ]
    
    # 更丰富的主题关键词
    themes = [
        "靶向治疗", "免疫治疗", "精准放疗", "早期筛查", "分子分型", 
        "耐药机制", "生物标志物", "基因检测", "液体活检", "肿瘤微环境",
        "联合治疗", "新辅助治疗", "辅助治疗", "姑息治疗", "生存质量",
        "副作用管理", "患者报告结局", "成本效益", "真实世界证据", "转化研究"
    ]
    
    # 更丰富的后缀描述
    suffixes = [
        "的临床实践", "的研究进展", "的突破性发现", "的多中心研究", "的系统评价",
        "的长期随访结果", "的荟萃分析", "的前瞻性研究", "的回顾性分析", "的专家共识",
        "的治疗策略", "的诊断价值", "的预后意义", "的机制研究", "的应用前景"
    ]
    
    data = {
        "article_id": range(1, num+1),
        "title": [],
        "type": np.random.choice(article_types, num),
        "cancer_type": [np.random.choice(cancer_types, np.random.randint(1, 3), replace=False).tolist() for _ in range(num)],
        "department": [np.random.choice(departments, np.random.randint(1, 3), replace=False).tolist() for _ in range(num)],
        "authority": np.random.choice(["国际顶级期刊", "国内核心期刊", "学会指南", "会议摘要", "预印本"], num, p=[0.1, 0.4, 0.2, 0.2, 0.1]),
        "impact_factor": np.random.uniform(0, 30, num).round(1),
        "pub_date": pd.date_range(start="2023-01-01", end="2024-05-01", periods=num),
        "keywords": [[np.random.choice(themes, np.random.randint(3, 6), replace=False).tolist()] for _ in range(num)]
    }
    
    # 生成多样化的标题
    for i in range(num):
        cancer = np.random.choice(cancer_types)
        theme = np.random.choice(themes)
        
        # 随机选择标题结构
        title_type = np.random.choice(["prefix", "simple", "question", "numbered"], p=[0.4, 0.3, 0.2, 0.1])
        
        if title_type == "prefix":
            prefix = np.random.choice(title_prefixes)
            title = f"{prefix}{cancer}{theme}{suffixes[i % len(suffixes)]}"
        elif title_type == "simple":
            title = f"{cancer}{theme}的最新研究进展"
        elif title_type == "question":
            questions = [
                f"{cancer}治疗：{theme}是否带来生存获益？",
                f"如何优化{cancer}的{theme}策略？",
                f"{theme}在{cancer}治疗中的价值几何？"
            ]
            title = np.random.choice(questions)
        else:  # numbered
            title = f"研究报道{(i+1)}：{theme}在{cancer}中的应用"
        
        data["title"].append(title)
    
    return pd.DataFrame(data)

# 生成数据
doctors = generate_doctors(100)
articles = generate_articles(200)

# ======================
# 2. 特征工程
# ======================
# 特征编码
def encode_features(doctors_df, articles_df):
    # 医生特征编码
    doctor_features = pd.DataFrame()
    
    # 职级编码
    level_map = {"住院医师": 1, "主治医师": 2, "副主任医师": 3, "主任医师": 4}
    doctor_features["level"] = doctors_df["level"].map(level_map).fillna(0)
    
    # 医院等级编码
    hospital_map = {"社区医院": 1, "二级医院": 2, "三乙医院": 3, "三甲医院": 4}
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
    
    # 工作经验标准化 - 添加缺失值处理
    if doctors_df["years_exp"].max() > 0:
        doctor_features["years_exp"] = doctors_df["years_exp"] / doctors_df["years_exp"].max()
    else:
        doctor_features["years_exp"] = 0  # 处理除零错误
    
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
    authority_map = {"会议摘要": 1, "国内核心期刊": 2, "学会指南": 3, "国际顶级期刊": 4}
    article_features["authority"] = articles_df["authority"].map(authority_map).fillna(0)
    
    # 影响因子标准化 - 添加缺失值处理
    max_impact = articles_df["impact_factor"].max()
    if max_impact > 0:
        article_features["impact_factor"] = articles_df["impact_factor"] / max_impact
    else:
        article_features["impact_factor"] = 0
    
    # 时间衰减因子 - 添加缺失值处理
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
    
    # 验证没有 NaN
    if doctor_features.isnull().any().any() or article_features.isnull().any().any():
        st.error("警告：特征矩阵中存在空值！")
        st.write("医生特征空值:", doctor_features.isnull().sum().sum())
        st.write("文章特征空值:", article_features.isnull().sum().sum())
    
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
    
    # 新增：验证向量没有 NaN
    if np.isnan(doctor_vec).any():
        st.error(f"医生向量包含NaN值: {doctor_idx}")
        st.write(doctor_vec)
        doctor_vec = np.nan_to_num(doctor_vec)
    
    # 获取文章特征矩阵
    article_vecs = article_features.values
    
    # 新增：验证矩阵没有 NaN
    if np.isnan(article_vecs).any():
        st.error("文章特征矩阵包含NaN值！")
        # 计算每列的NaN数量
        nan_counts = np.isnan(article_vecs).sum(axis=0)
        st.write("每列NaN数量:", nan_counts)
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
st.sidebar.header("👨‍⚕️ 医生信息设置")
selected_doctor = st.sidebar.selectbox("选择医生档案", doctors["name"])

# 获取选定医生的信息
doctor_info = doctors[doctors["name"] == selected_doctor].iloc[0]

# 显示医生信息
st.sidebar.subheader("医生档案详情")
st.sidebar.markdown(f"**职级:** {doctor_info['level']}")
st.sidebar.markdown(f"**医院等级:** {doctor_info['hospital_level']}")
st.sidebar.markdown(f"**科室:** {doctor_info['department']}")
st.sidebar.markdown(f"**擅长瘤肿:** {', '.join(doctor_info['specialty'])}")  
st.sidebar.markdown(f"**工作经验:** {doctor_info['years_exp']}年")

# 推荐设置
st.sidebar.subheader("推荐设置")
top_n = st.sidebar.slider("推荐文章数量", 3, 10, 5)
weight_specialty = st.sidebar.slider("擅长瘤肿权重", 0.0, 1.0, 0.6)
weight_department = st.sidebar.slider("科室权重", 0.0, 1.0, 0.3)
weight_level = st.sidebar.slider("职级权重", 0.0, 1.0, 0.1)

# 调整权重
doctor_features_weighted = doctor_features.copy()
doctor_features_weighted[cancer_cols] *= weight_specialty
doctor_features_weighted[dept_cols] *= weight_department
doctor_features_weighted["level"] *= weight_level

# 主界面
st.header(f"📰 为 {selected_doctor} 推荐的肿瘤资讯")

# 添加进度条
progress_bar = st.progress(0)
status_text = st.empty()

# 模拟计算过程
for i in range(100):
    progress_bar.progress(i + 1)
    status_text.text(f"正在匹配最佳资讯... {i+1}%")
    time.sleep(0.01)

# 获取推荐结果
recommended_articles = recommend_articles(doctor_info["doctor_id"], top_n)

# 显示推荐结果
for idx, row in recommended_articles.iterrows():
    with st.expander(f"**[{row['article_id']}] {row['title']}** (匹配度: {row['match_score']:.2f})", expanded=idx==0):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # 文章类型图标
            type_icon = {
                "临床研究": "🔬",
                "指南更新": "📜",
                "病例讨论": "📋",
                "综述": "📚",
                "临床试验": "💊",
                "技术进展": "⚙️",
                "专家共识": "👥",
                "研究简报": "📰",
                "治疗新策略": "💡",
                "药物评价": "💊",
                "诊断突破": "🔍",
                "预后模型": "📊",
                "多中心研究": "🌐"
            }.get(row["type"], "📄")
            
            st.markdown(f"**类型:** {type_icon} {row['type']}")
            st.markdown(f"**相关瘤肿:** {', '.join(row['cancer_type'])}")
            st.markdown(f"**相关科室:** {', '.join(row['department'])}")
            st.markdown(f"**发表来源:** {row['authority']}")
            st.markdown(f"**影响因子:** {row['impact_factor']}")
            st.markdown(f"**发布日期:** {row['pub_date'].strftime('%Y-%m-%d')}")
        
        with col2:
            # 生成匹配理由
            match_reasons = []
            
            # 科室匹配
            if any(dept in row["department"] for dept in [doctor_info["department"]]):
                match_reasons.append(f"与您的科室({doctor_info['department']})相关")
            
            # 瘤肿匹配
            common_cancers = set(doctor_info["specialty"]) & set(row["cancer_type"])
            if common_cancers:
                match_reasons.append(f"涉及您擅长的{', '.join(common_cancers)}")
            
            # 职级匹配
            if doctor_info["level"] in ["副主任医师", "主任医师"] and row["type"] in ["指南更新", "专家共识", "综述"]:
                match_reasons.append("符合您的高级职称需求")
            elif doctor_info["level"] in ["住院医师", "主治医师"] and row["type"] in ["病例讨论", "临床研究", "治疗新策略"]:
                match_reasons.append("符合您的临床学习需求")
            
            # 影响因子
            if row["impact_factor"] > 15:
                match_reasons.append("来自高影响力期刊")
            
            # 显示匹配理由
            if match_reasons:
                st.info("**匹配理由:** " + "；".join(match_reasons))
            else:
                st.info("**匹配理由:** 综合特征匹配")
            
            # 显示关键词
            keywords = row["keywords"][0] if isinstance(row["keywords"], list) else row["keywords"]
            st.markdown("**关键词:** " + ", ".join(keywords))
            
            # 生成摘要 - 更专业的医学摘要
            cancer = np.random.choice(row["cancer_type"])
            keywords_str = ", ".join(keywords[:3])
            st.markdown(f"""
            **摘要:**  
            本研究探讨了{keywords_str}在{cancer}治疗中的应用。通过对{np.random.randint(50, 500)}例患者的分析，
            发现{np.random.choice(["显著提高生存率", "明显改善生活质量", "有效降低复发风险", "具有良好安全性"])}。
            研究结果为{cancer}的临床实践提供了新的循证依据。
            """)

# 添加分隔线
st.markdown("---")

# 推荐分析图表 (英文版)
st.subheader("Recommendation Analysis")
col1, col2 = st.columns(2)

with col1:
    # 文章类型分布 (英文)
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
        st.error(f"Error drawing type distribution: {str(e)}")
        st.write("Distribution of Recommended Article Types:")
        st.dataframe(type_counts)

with col2:
    # 匹配分数分布 (英文)
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
        st.error(f"Error drawing score distribution: {str(e)}")
        st.write("Match Score Statistics:")
        st.write(recommended_articles["match_score"].describe())

# 系统说明
st.markdown("---")
st.subheader("系统说明")
st.markdown("""
本推荐系统基于医生以下特征进行个性化推荐：
1. **职级**：不同职级医生关注内容不同（住院医师更关注基础知识，主任医师更关注前沿进展）
2. **医院等级**：不同级别医院可获取资源不同
3. **科室**：精准匹配医生所在科室相关内容
4. **擅长瘤肿**：重点推荐医生专长领域的资讯
5. **工作经验**：经验丰富的医生更关注复杂病例和前沿研究

系统通过特征向量化和余弦相似度算法计算匹配度，权重可调。
""")

# 在侧边栏添加调试选项
debug_mode = st.sidebar.checkbox("调试模式")

if debug_mode:
    st.sidebar.subheader("调试信息")
    st.sidebar.write("医生特征维度:", doctor_features.shape)
    st.sidebar.write("文章特征维度:", article_features.shape)
    
    # 显示NaN数量
    doctor_nan = doctor_features.isnull().sum().sum()
    article_nan = article_features.isnull().sum().sum()
    st.sidebar.write(f"医生特征NaN数量: {doctor_nan}")
    st.sidebar.write(f"文章特征NaN数量: {article_nan}")
    
    if doctor_nan > 0:
        st.sidebar.write("医生特征NaN列:")
        st.sidebar.write(doctor_features.isnull().sum())
    
    if article_nan > 0:
        st.sidebar.write("文章特征NaN列:")
        st.sidebar.write(article_features.isnull().sum())

# 隐藏进度条
progress_bar.empty()
status_text.empty()
