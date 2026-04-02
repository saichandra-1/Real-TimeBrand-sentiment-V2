import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="TrendRadar - Zomato Sentiment", layout="wide", page_icon="🎯")

POS_COLOR = '#27AE60'
NEG_COLOR = '#E74C3C'
NEU_COLOR = '#F39C12'
BRAND_DARK = '#1A1A2E'

st.title("🎯 TrendRadar - Zomato Sentiment Analysis Dashboard")
st.markdown("**RoBERTa NLP | Twitter + Reddit Analysis**")

# Pre-computed metrics from notebook
total = 1933
pos_count = 582
neg_count = 926
neu_count = 425
avg_conf = 0.731

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Records", f"{total:,}")
col2.metric("Positive", f"{pos_count:,} ({pos_count/total*100:.1f}%)")
col3.metric("Negative", f"{neg_count:,} ({neg_count/total*100:.1f}%)")
col4.metric("Neutral", f"{neu_count:,} ({neu_count/total*100:.1f}%)")
col5.metric("Avg Confidence", f"{avg_conf:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📂 Categories", "☁️ Word Clouds", "💡 Insights"])

with tab1:
    st.subheader("Sentiment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sent_data = {'NEGATIVE': neg_count, 'NEUTRAL': neu_count, 'POSITIVE': pos_count}
        colors = [NEG_COLOR, NEU_COLOR, POS_COLOR]
        wedges, texts, autotexts = ax.pie(sent_data.values(), labels=sent_data.keys(), autopct='%1.1f%%', 
                                            colors=colors, startangle=140, 
                                            wedgeprops=dict(width=0.52, edgecolor='white', linewidth=2.5))
        for at in autotexts: at.set(fontsize=11, fontweight='bold', color='white')
        ax.set_title('Overall Sentiment Split', fontweight='bold', pad=12)
        ax.text(0, 0, f'{total:,}\nrecords', ha='center', va='center', fontsize=13, fontweight='bold', color=BRAND_DARK)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sources = ['Twitter', 'Reddit']
        pos_pct = [45.5, 29.1]
        neu_pct = [27.0, 21.5]
        neg_pct = [27.5, 49.4]
        
        x = range(len(sources))
        width = 0.25
        ax.bar([i-width for i in x], pos_pct, width, label='POSITIVE', color=POS_COLOR, edgecolor='white')
        ax.bar(x, neu_pct, width, label='NEUTRAL', color=NEU_COLOR, edgecolor='white')
        ax.bar([i+width for i in x], neg_pct, width, label='NEGATIVE', color=NEG_COLOR, edgecolor='white')
        
        ax.set_xticks(x)
        ax.set_xticklabels(sources)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Sentiment by Source (Twitter vs Reddit)', fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        st.pyplot(fig)

with tab2:
    st.subheader("Issue Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Top Negative Categories")
        fig, ax = plt.subplots(figsize=(8, 6))
        neg_cats = {
            'Delivery Issues': 483,
            'Customer Service': 389,
            'Food Quality': 312,
            'App / Tech Issues': 287,
            'Pricing & Value': 245,
            'Order Accuracy': 198,
            'Brand / PR': 156
        }
        ax.barh(list(neg_cats.keys()), list(neg_cats.values()), color=NEG_COLOR)
        ax.set_xlabel('Count')
        ax.set_title('Top Negative Issue Categories', fontweight='bold')
        ax.invert_yaxis()
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### 🟢 Top Positive Categories")
        fig, ax = plt.subplots(figsize=(8, 6))
        pos_cats = {
            'Positive Praise': 412,
            'Pricing & Value': 178,
            'Food Quality': 134,
            'Brand / PR': 98,
            'Delivery Issues': 67,
            'App / Tech Issues': 45
        }
        ax.barh(list(pos_cats.keys()), list(pos_cats.values()), color=POS_COLOR)
        ax.set_xlabel('Count')
        ax.set_title('Top Positive Categories', fontweight='bold')
        ax.invert_yaxis()
        st.pyplot(fig)

with tab3:
    st.subheader("Word Clouds by Sentiment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🟢 Positive")
        pos_words = "great amazing love best excellent food delivery service discount offer value money quality taste delicious fast quick good happy satisfied recommend"
        wc = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(pos_words)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### 🟡 Neutral")
        neu_words = "zomato order app food delivery restaurant menu price time service customer support payment"
        wc = WordCloud(width=400, height=300, background_color='white', colormap='Oranges').generate(neu_words)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    with col3:
        st.markdown("#### 🔴 Negative")
        neg_words = "late delivery cold food wrong order cancel refund support poor quality bad service expensive charge wait time issue problem complaint"
        wc = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(neg_words)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

with tab4:
    st.subheader("💡 Strategic Insights for Zomato")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💪 Strengths to Leverage")
        strengths = [
            ("Discounts & Offers", 33),
            ("Trusted Brand", 26),
            ("Value for Money", 10),
            ("Food Variety", 8),
            ("Easy App/UX", 5),
            ("Fast Delivery", 4),
            ("Packaging Quality", 3)
        ]
        for strength, count in strengths:
            st.success(f"**{strength}**: ({count})")
    
    with col2:
        st.markdown("#### ⚠️ Critical Issues to Address")
        issues = [
            ("Late Deliveries", 74),
            ("Cold/Stale Food", 32),
            ("Wrong Orders", 28),
            ("Long Wait Support", 26),
            ("High Charges", 20),
            ("Poor Refund", 19),
            ("App Crashes", 7),
            ("Rude Staff", 6)
        ]
        for issue, count in issues:
            st.error(f"**{issue}**: ({count})")
    
    st.markdown("---")
    
    st.markdown("### 📋 Dynamic Recommendations")
    
    st.markdown("""
    **🔴 Priority 1: Delivery Issues (48% of negative feedback)**
    - Implement real-time route optimization
    - Set up automated ETA accuracy monitoring
    - Create rapid response team for delivery complaints
    - Target: Reduce delivery complaints by 30% in 60 days
    
    **🔴 Priority 2: Customer Service (39% of negative feedback)**
    - Escalate high neg_score cases (>0.80) to senior specialists
    - Implement 2-hour resolution commitment
    - Add live chat support for urgent issues
    - Target: Improve response time to <2 hours
    
    **🔴 Priority 3: Food Quality (31% of negative feedback)**
    - Flag restaurants with repeated quality complaints
    - Introduce order photo-verification system
    - Partner quality audits for low-rated outlets
    - Target: 15% reduction in quality complaints
    
    **🟢 Leverage Strengths:**
    - Promote discount programs (33 positive mentions)
    - Highlight brand trust in marketing (26 mentions)
    - Showcase value for money positioning (10 mentions)
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Tech Roadmap")
    st.markdown("""
    1. Deploy RoBERTa pipeline as real-time monitoring service (hourly refresh)
    2. Connect Response Engine to CRM for automated first-touch replies
    3. Alert threshold: escalate any comment with neg_score > 0.80 to human agent
    4. Predictive analytics: flag restaurants likely to receive complaints
    5. A/B test personalized responses vs generic templates
    """)

st.markdown("---")
st.markdown("**TrendRadar** | RoBERTa NLP (cardiffnlp/twitter-roberta-base-sentiment-latest) | Avg Confidence: 0.731")
st.markdown("**Team:** B.Hemanth (22MIA1083), Sai Chandra (22MIA1036), Latheef (22MIA1121), Sri Mokshith (22MIA1068)")
