import streamlit as st
import pandas as pd
import joblib
import pickle
import re
import plotly.express as px
import plotly.graph_objects as go

# 1. Core Configuration & Modern Styling
st.set_page_config(
    page_title="Email Intelligence Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #007bff;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .hero-header {
        text-align: center;
        padding: 40px 0;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Email Intelligence Dashboard</div>
    <div class="hero-subtitle">Advanced Machine Learning for Spam Detection & Analysis</div>
</div>
""", unsafe_allow_html=True)

# 2. Load Analytics Artifacts
@st.cache_resource
def load_models():
    model = joblib.load('spam_model.pkl')
    with open('model_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

@st.cache_data
def load_data():
    df = pd.read_csv('emails.csv')
    if 'Email No.' in df.columns:
        df = df.drop(columns=['Email No.'])
    return df

try:
    model, features = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Prepare Data for visualisations
# Map prediction 0/1 to Ham/Spam
if 'Class' not in df.columns:
    df['Class'] = df['Prediction'].map({1: 'Spam', 0: 'Ham'})

# Calculate total word count per email
word_cols = df.columns.drop(['Prediction', 'Class'], errors='ignore')
if 'Total_Words' not in df.columns:
    df['Total_Words'] = df[word_cols].sum(axis=1)

# 3. Modern Dashboard Layout
tab1, tab2 = st.tabs(["Email Classification", "Model & Data Insights"])

# 5. 'Email Classification' Section (Prediction UI)
with tab1:
    st.markdown("### Predict New Email")
    st.markdown("Paste the content of an email below to analyze its spam probability.")
    
    with st.container():
        user_input = st.text_area("Email Content", height=200, placeholder="Paste your email text here...")
        predict_button = st.button("Analyze Email", type="primary", use_container_width=True)
        
    if predict_button and user_input:
        # Preprocessing Engine
        # Clean the input text
        text = re.sub(r'<[^>]+>', '', user_input)  # Strip HTML
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = text.split()
        
        # Map to model_features
        word_counts = {}
        for word in words:
            if word in features:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create 1-row DataFrame
        input_data = {feat: [word_counts.get(feat, 0)] for feat in features}
        input_df = pd.DataFrame(input_data)
        
        # Make Prediction
        prediction = model.predict(input_df)[0]
        
        # Output Hero
        st.markdown("---")
        if prediction == 0:
            st.success("✅ SAFE EMAIL (HAM)")
        else:
            st.error("🚨 SPAM DETECTED")

# 4. 'Model & Data Insights' Section (Analytics)
with tab2:
    st.markdown("### Model Performance")
    
    # Row 1 - Key Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">97.14%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Precision (Spam)</div>
            <div class="metric-value">94.00%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Recall (Spam)</div>
            <div class="metric-value">96.00%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">F1-Score (Spam)</div>
            <div class="metric-value">95.00%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### Advanced Data Discovery Suite")
    
    # Row 2 - Class Distribution & Top 20 Words
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.markdown("#### 1. Class Distribution")
        class_counts = df['Class'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        
        fig_pie = px.pie(class_counts, values='Count', names='Class', 
                         color='Class', color_discrete_map={'Spam': '#dc3545', 'Ham': '#28a745'},
                         hole=0.5)
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with row2_col2:
        st.markdown("#### 2. Top 20 Feature Dominance")
        word_freq = df[word_cols].sum().sort_values(ascending=False).head(20).reset_index()
        word_freq.columns = ['Word', 'Frequency']
        
        fig_bar = px.bar(word_freq, x='Frequency', y='Word', orientation='h',
                         color='Frequency', color_continuous_scale='Plasma')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    
    # Row 3 - Histogram & Correlation Heatmap
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        st.markdown("#### 3. Word Length Distribution")
        fig_hist = px.histogram(df, x="Total_Words", color="Class", 
                                color_discrete_map={'Spam': '#dc3545', 'Ham': '#28a745'},
                                barmode="overlay", nbins=100)
        fig_hist.update_layout(xaxis_title="Total Words in Email", yaxis_title="Count of Emails",
                               margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with row3_col2:
        st.markdown("#### 4. Keyword Correlation Heatmap")
        st.markdown("Correlation of highly predictive words with the Spam class.")
        key_words = ['money', 'click', 'enron', 'deal']
        # Filter existing keywords to avoid KeyError
        valid_keywords = [w for w in key_words if w in df.columns]
        
        if valid_keywords:
            corr_data = df[valid_keywords + ['Prediction']].corr()
            fig_heatmap = px.imshow(corr_data, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig_heatmap.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("Selected keywords are not present in the dataset vocabulary.")
            
    st.markdown("---")
    
    # Row 4 - Average Feature Intensity
    st.markdown("#### 5. Feature Intensity Comparison")
    st.markdown("Analyze the distribution of specific high-frequency words across Spam and Ham emails.")
    
    # Find common words for dropdown
    common_words_options = ['free', 'thanks', 'meeting', 'money', 'business', 'email', 'deal', 'click']
    valid_common_words = [w for w in common_words_options if w in df.columns]
    
    if valid_common_words:
        selected_word = st.selectbox("Select a keyword to analyze:", valid_common_words)
        
        fig_box = px.violin(df, x="Class", y=selected_word, color="Class", 
                          color_discrete_map={'Spam': '#dc3545', 'Ham': '#28a745'},
                          box=True, points=False)
        fig_box.update_layout(xaxis_title="Email Class", yaxis_title=f"Frequency of '{selected_word}'",
                              margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("No default common words found in the vocabulary.")
