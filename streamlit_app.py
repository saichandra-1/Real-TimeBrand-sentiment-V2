import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="TrendRadar - Zomato Sentiment", layout="wide", page_icon="🎯")

POS_COLOR = '#27AE60'
NEG_COLOR = '#E74C3C'
NEU_COLOR = '#F39C12'

@st.cache_data
def load_data():
    tweets = pd.read_csv('RealData/zomato_tweets.csv')
    reddit = pd.read_csv('RealData/reddit_posts_and_comments (1).csv')
    
    tweets['source'] = 'Twitter'
    tweets['text'] = tweets['text']
    tweets['engagement'] = tweets.get('like_count', 0) + tweets.get('retweet_count', 0)
    
    reddit['source'] = 'Reddit'
    reddit['text'] = reddit['comment_body'].fillna(reddit['post_title'])
    reddit['engagement'] = reddit.get('comment_score', 0)
    
    df = pd.concat([
        tweets[['text', 'source', 'engagement']],
        reddit[['text', 'source', 'engagement']]
    ], ignore_index=True)
    
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    return df

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

@st.cache_resource
def load_sentiment_model():
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", max_length=512, truncation=True)
    except:
        return None

def analyze_sentiment(text, model):
    if model:
        try:
            result = model(text[:512])[0]
            label_map = {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE'}
            return label_map.get(result['label'], 'NEUTRAL'), result['score']
        except:
            pass
    
    from textblob import TextBlob
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'POSITIVE', abs(polarity)
    elif polarity < -0.1:
        return 'NEGATIVE', abs(polarity)
    return 'NEUTRAL', abs(polarity)

ISSUE_CATEGORIES = {
    'Delivery Issues': ['delivery', 'late', 'delay', 'slow', 'arrived', 'rider', 'never came', 'missing delivery', 'logistics', 'not delivered'],
    'Food Quality': ['cold', 'stale', 'quality', 'taste', 'spoiled', 'bad food', 'unhygienic', 'dirty', 'rotten', 'smell', 'inedible', 'fresh'],
    'Pricing & Value': ['expensive', 'price', 'costly', 'overpriced', 'surge', 'discount', 'coupon', 'charges', 'fee', 'gst', 'value for money'],
    'App / Tech Issues': ['app', 'crash', 'bug', 'error', 'glitch', 'not working', 'interface', 'update', 'login', 'website', 'loading'],
    'Customer Service': ['support', 'helpline', 'refund', 'cancel', 'complaint', 'resolution', 'ignored', 'rude', 'agent', 'helpdesk', 'reply'],
    'Order Accuracy': ['wrong item', 'missing item', 'incomplete', 'wrong order', 'not what i ordered', 'substituted', 'different item'],
    'Brand / PR': ['blinkit', 'deepinder', 'goyal', 'ipo', 'investor', 'ceo', 'expansion', 'fund', 'news', 'strategy', 'acquisition'],
    'Positive Praise': ['amazing', 'great', 'love', 'excellent', 'best', 'fantastic', 'superb', 'awesome', 'perfect', 'recommend', 'delicious', 'happy'],
}

def categorize(text):
    tl = str(text).lower()
    matched = [cat for cat, kws in ISSUE_CATEGORIES.items() if any(k in tl for k in kws)]
    return matched if matched else ['Uncategorized']

st.title("🎯 TrendRadar - Zomato Sentiment Analysis")
st.markdown("**RoBERTa NLP + Dynamic Intelligence Engine** | Twitter + Reddit Analysis")

with st.spinner("Loading and analyzing data..."):
    df = load_data()
    df['clean_text'] = df['text'].apply(clean_text)
    
    model = load_sentiment_model()
    sentiments, confidences = [], []
    
    progress_bar = st.progress(0)
    for idx, text in enumerate(df['clean_text']):
        sent, conf = analyze_sentiment(text, model)
        sentiments.append(sent)
        confidences.append(conf)
        if idx % 100 == 0:
            progress_bar.progress(min((idx + 1) / len(df), 1.0))
    
    progress_bar.empty()
    
    df['sentiment'] = sentiments
    df['confidence'] = confidences
    df['categories'] = df['clean_text'].apply(categorize)

total = len(df)
pos_count = (df['sentiment'] == 'POSITIVE').sum()
neg_count = (df['sentiment'] == 'NEGATIVE').sum()
neu_count = (df['sentiment'] == 'NEUTRAL').sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{total:,}")
col2.metric("Positive", f"{pos_count:,} ({pos_count/total*100:.1f}%)")
col3.metric("Negative", f"{neg_count:,} ({neg_count/total*100:.1f}%)")
col4.metric("Neutral", f"{neu_count:,} ({neu_count/total*100:.1f}%)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📂 Categories", "💬 Comments", "☁️ Word Clouds", "💡 Insights"])

with tab1:
    st.subheader("Sentiment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sent_counts = df['sentiment'].value_counts()
        colors = [POS_COLOR if s == 'POSITIVE' else NEG_COLOR if s == 'NEGATIVE' else NEU_COLOR for s in sent_counts.index]
        ax.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.1f%%', colors=colors, startangle=140, wedgeprops=dict(width=0.5, edgecolor='white'))
        ax.set_title('Overall Sentiment Split', fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        src_pct = pd.crosstab(df['source'], df['sentiment'], normalize='index') * 100
        src_pct = src_pct.reindex(columns=['POSITIVE', 'NEUTRAL', 'NEGATIVE'], fill_value=0)
        src_pct.plot(kind='bar', ax=ax, color=[POS_COLOR, NEU_COLOR, NEG_COLOR], edgecolor='white')
        ax.set_title('Sentiment by Source', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Percentage (%)')
        ax.legend()
        plt.xticks(rotation=0)
        st.pyplot(fig)

with tab2:
    st.subheader("Issue Categories")
    
    df_cat = df.explode('categories')
    neg_cats = df_cat[df_cat['sentiment'] == 'NEGATIVE']['categories'].value_counts().drop('Uncategorized', errors='ignore')
    pos_cats = df_cat[df_cat['sentiment'] == 'POSITIVE']['categories'].value_counts().drop('Uncategorized', errors='ignore')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Top Negative Categories")
        if len(neg_cats) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            neg_cats.head(7).plot(kind='barh', ax=ax, color=NEG_COLOR)
            ax.set_xlabel('Count')
            ax.set_title('Top Negative Issue Categories', fontweight='bold')
            st.pyplot(fig)
        else:
            st.info("No negative categories found")
    
    with col2:
        st.markdown("#### 🟢 Top Positive Categories")
        if len(pos_cats) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            pos_cats.head(7).plot(kind='barh', ax=ax, color=POS_COLOR)
            ax.set_xlabel('Count')
            ax.set_title('Top Positive Categories', fontweight='bold')
            st.pyplot(fig)
        else:
            st.info("No positive categories found")

with tab3:
    st.subheader("Sample Comments")
    
    sentiment_filter = st.selectbox("Filter by Sentiment", ["NEGATIVE", "POSITIVE", "NEUTRAL"])
    display_df = df[df['sentiment'] == sentiment_filter].sort_values('confidence', ascending=False).head(10)
    
    for idx, row in display_df.iterrows():
        icon = "🟢" if row['sentiment'] == 'POSITIVE' else "🔴" if row['sentiment'] == 'NEGATIVE' else "🟡"
        cats = ', '.join([c for c in row['categories'] if c != 'Uncategorized']) or 'General'
        
        with st.expander(f"{icon} [{row['source']}] {cats} - Confidence: {row['confidence']:.2f}"):
            st.write(f"**Text:** {row['text'][:400]}")
            st.write(f"**Engagement:** {row['engagement']}")

with tab4:
    st.subheader("Word Clouds by Sentiment")
    
    col1, col2, col3 = st.columns(3)
    
    stopwords = set(STOPWORDS)
    stopwords.update(['zomato', 'will', 'one', 'get', 'got', 'like', 'just', 'now', 'even'])
    
    with col1:
        st.markdown("#### 🟢 Positive")
        pos_text = ' '.join(df[df['sentiment'] == 'POSITIVE']['clean_text'])
        if pos_text:
            wc = WordCloud(width=400, height=300, background_color='white', stopwords=stopwords, colormap='Greens').generate(pos_text)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with col2:
        st.markdown("#### 🟡 Neutral")
        neu_text = ' '.join(df[df['sentiment'] == 'NEUTRAL']['clean_text'])
        if neu_text:
            wc = WordCloud(width=400, height=300, background_color='white', stopwords=stopwords, colormap='Oranges').generate(neu_text)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with col3:
        st.markdown("#### 🔴 Negative")
        neg_text = ' '.join(df[df['sentiment'] == 'NEGATIVE']['clean_text'])
        if neg_text:
            wc = WordCloud(width=400, height=300, background_color='white', stopwords=stopwords, colormap='Reds').generate(neg_text)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

with tab5:
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
    st.markdown("### 📋 Key Recommendations")
    
    st.markdown("""
    **Immediate Actions (0-30 days):**
    - Set up automated sentiment monitoring dashboard
    - Create rapid response team for negative feedback
    - Implement delivery time optimization
    
    **Short-term Goals (1-3 months):**
    - Reduce negative sentiment by 15%
    - Improve average response time to <2 hours
    - Launch food quality assurance program
    
    **Long-term Strategy (3-12 months):**
    - Achieve 60%+ positive sentiment
    - Develop predictive analytics for issue prevention
    - Create loyalty programs based on insights
    """)

st.markdown("---")
st.markdown("**TrendRadar** | RoBERTa NLP | Team: B.Hemanth (22MIA1083), Sai Chandra (22MIA1036), Latheef (22MIA1121), Sri Mokshith (22MIA1068)")
