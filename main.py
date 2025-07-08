#!/usr/bin/env python3
"""
Smart News Analyzer
A sophisticated Python project that fetches, analyzes, and visualizes news sentiment
with ML-powered insights and interactive dashboard.

Features:
- Real-time news fetching from multiple sources
- Sentiment analysis using natural language processing
- Topic modeling and keyword extraction
- Interactive web dashboard
- Data visualization and export capabilities
- Automated reporting with email notifications
"""

import requests
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Third-party libraries (install with: pip install -r requirements.txt)
try:
    from textblob import TextBlob
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import plotly.graph_objects as go
    import plotly.express as px
    from flask import Flask, render_template, jsonify
    import schedule
    import time
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install textblob wordcloud matplotlib seaborn scikit-learn plotly flask schedule")

@dataclass
class NewsArticle:
    """Data class for news articles"""
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    keywords: List[str] = None

class NewsAnalyzer:
    """Main class for news analysis system"""
    
    def __init__(self, db_path: str = "news_analysis.db"):
        self.db_path = db_path
        self.setup_database()
        self.setup_logging()
        
    def setup_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                url TEXT UNIQUE,
                source TEXT,
                published_at TIMESTAMP,
                sentiment_score REAL,
                sentiment_label TEXT,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date DATE,
                total_articles INTEGER,
                avg_sentiment REAL,
                top_keywords TEXT,
                report_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('news_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_news(self, api_key: str, sources: List[str] = None, 
                  query: str = None, days_back: int = 1) -> List[NewsArticle]:
        """
        Fetch news from News API
        Get your free API key from: https://newsapi.org/
        """
        if not sources:
            sources = ['bbc-news', 'reuters', 'associated-press', 'cnn', 'the-guardian-uk']
        
        articles = []
        base_url = "https://newsapi.org/v2/everything"
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'apiKey': api_key,
            'sources': ','.join(sources),
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': 100
        }
        
        if query:
            params['q'] = query
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('articles', []):
                if item['content'] and item['content'] != '[Removed]':
                    article = NewsArticle(
                        title=item['title'],
                        content=item['content'],
                        url=item['url'],
                        source=item['source']['name'],
                        published_at=datetime.fromisoformat(item['publishedAt'].replace('Z', '+00:00'))
                    )
                    articles.append(article)
            
            self.logger.info(f"Fetched {len(articles)} articles")
            return articles
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching news: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
            
            return polarity, label
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return 0.0, "neutral"
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords using simple frequency analysis"""
        try:
            # Clean text
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            # Common stop words
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
                'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'must', 'can', 'said', 'says', 'news', 'report',
                'article', 'according', 'also', 'new', 'first', 'last', 'one', 'two',
                'get', 'got', 'go', 'went', 'going', 'come', 'came', 'coming',
                'make', 'made', 'making', 'take', 'took', 'taking', 'see', 'saw',
                'seen', 'know', 'knew', 'known', 'think', 'thought', 'thinking',
                'people', 'person', 'man', 'woman', 'way', 'time', 'year', 'day',
                'home', 'world', 'life', 'hand', 'part', 'child', 'eye', 'woman',
                'place', 'work', 'week', 'case', 'point', 'government', 'company'
            }
            
            # Split into words and filter
            words = text.split()
            filtered_words = [
                word for word in words 
                if len(word) >= 3 and word not in stop_words
            ]
            
            # Count word frequencies
            word_freq = Counter(filtered_words)
            
            # Get top keywords
            top_keywords = [word for word, count in word_freq.most_common(num_keywords)]
            
            return top_keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
    
    def process_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Process articles with sentiment analysis and keyword extraction"""
        processed_articles = []
        
        for article in articles:
            try:
                # Analyze sentiment
                full_text = f"{article.title} {article.content}"
                sentiment_score, sentiment_label = self.analyze_sentiment(full_text)
                article.sentiment_score = sentiment_score
                article.sentiment_label = sentiment_label
                
                # Extract keywords
                article.keywords = self.extract_keywords(full_text)
                
                processed_articles.append(article)
                
            except Exception as e:
                self.logger.error(f"Error processing article: {e}")
                continue
        
        return processed_articles
    
    def save_articles(self, articles: List[NewsArticle]):
        """Save articles to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO articles 
                    (title, content, url, source, published_at, sentiment_score, 
                     sentiment_label, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.title,
                    article.content,
                    article.url,
                    article.source,
                    article.published_at.isoformat(),
                    article.sentiment_score,
                    article.sentiment_label,
                    ','.join(article.keywords) if article.keywords else ''
                ))
            except sqlite3.IntegrityError:
                # Article already exists
                continue
        
        conn.commit()
        conn.close()
        self.logger.info(f"Saved {len(articles)} articles to database")
    
    def generate_visualizations(self, output_dir: str = "visualizations"):
        """Generate various visualizations"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Get data from database
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM articles 
            WHERE created_at >= datetime('now', '-7 days')
        """, conn)
        conn.close()
        
        if df.empty:
            self.logger.warning("No data available for visualization")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Sentiment Distribution
        plt.figure(figsize=(10, 6))
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=[colors.get(x, 'blue') for x in sentiment_counts.index])
        plt.title('News Sentiment Distribution (Last 7 Days)')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sentiment_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sentiment Timeline
        df['published_at'] = pd.to_datetime(df['published_at'])
        daily_sentiment = df.groupby([df['published_at'].dt.date, 'sentiment_label']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 6))
        daily_sentiment.plot(kind='area', stacked=True, alpha=0.7, 
                           color=['green', 'gray', 'red'])
        plt.title('Daily Sentiment Trends')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.legend(title='Sentiment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sentiment_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Source Analysis
        plt.figure(figsize=(10, 6))
        source_sentiment = df.groupby(['source', 'sentiment_label']).size().unstack(fill_value=0)
        source_sentiment.plot(kind='bar', stacked=True, color=['green', 'gray', 'red'])
        plt.title('Sentiment by News Source')
        plt.xlabel('News Source')
        plt.ylabel('Number of Articles')
        plt.legend(title='Sentiment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/source_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Word Cloud
        all_keywords = []
        for keywords_str in df['keywords'].dropna():
            all_keywords.extend(keywords_str.split(','))
        
        if all_keywords:
            keyword_freq = Counter(all_keywords)
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                colormap='viridis').generate_from_frequencies(keyword_freq)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Frequent Keywords (Last 7 Days)')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/wordcloud.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Visualizations saved to {output_dir}/")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM articles 
            WHERE created_at >= datetime('now', '-7 days')
        """, conn)
        conn.close()
        
        if df.empty:
            return {"error": "No data available for report"}
        
        # Calculate metrics
        total_articles = len(df)
        avg_sentiment = df['sentiment_score'].mean()
        sentiment_distribution = df['sentiment_label'].value_counts().to_dict()
        
        # Top keywords
        all_keywords = []
        for keywords_str in df['keywords'].dropna():
            all_keywords.extend(keywords_str.split(','))
        top_keywords = Counter(all_keywords).most_common(10)
        
        # Most active sources
        top_sources = df['source'].value_counts().head(5).to_dict()
        
        # Sentiment by source
        source_sentiment = df.groupby('source')['sentiment_score'].mean().sort_values(ascending=False).to_dict()
        
        report = {
            "report_date": datetime.now().strftime('%Y-%m-%d'),
            "total_articles": total_articles,
            "average_sentiment": round(avg_sentiment, 3),
            "sentiment_distribution": sentiment_distribution,
            "top_keywords": top_keywords,
            "top_sources": top_sources,
            "source_sentiment": source_sentiment,
            "analysis_period": "Last 7 days"
        }
        
        # Save report to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_reports 
            (report_date, total_articles, avg_sentiment, top_keywords, report_data)
            VALUES (?, ?, ?, ?, ?)
        """, (
            report["report_date"],
            total_articles,
            avg_sentiment,
            ','.join([kw for kw, _ in top_keywords]),
            json.dumps(report)
        ))
        conn.commit()
        conn.close()
        
        return report
    
    def run_analysis_pipeline(self, api_key: str, sources: List[str] = None, 
                            query: str = None, generate_viz: bool = True):
        """Run complete analysis pipeline"""
        self.logger.info("Starting news analysis pipeline...")
        
        # Fetch news
        articles = self.fetch_news(api_key, sources, query)
        if not articles:
            self.logger.warning("No articles fetched")
            return
        
        # Process articles
        processed_articles = self.process_articles(articles)
        
        # Save to database
        self.save_articles(processed_articles)
        
        # Generate visualizations
        if generate_viz:
            self.generate_visualizations()
        
        # Generate report
        report = self.generate_report()
        
        self.logger.info("Analysis pipeline completed successfully")
        return report

# Flask Web Dashboard
app = Flask(__name__)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/report')
def get_report():
    """API endpoint for latest report"""
    analyzer = NewsAnalyzer()
    report = analyzer.generate_report()
    return jsonify(report)

@app.route('/api/articles')
def get_articles():
    """API endpoint for recent articles"""
    analyzer = NewsAnalyzer()
    conn = sqlite3.connect(analyzer.db_path)
    df = pd.read_sql_query("""
        SELECT title, source, sentiment_label, sentiment_score, published_at
        FROM articles 
        WHERE created_at >= datetime('now', '-7 days')
        ORDER BY published_at DESC
        LIMIT 50
    """, conn)
    conn.close()
    
    return jsonify(df.to_dict('records'))

def scheduled_analysis():
    """Function to run scheduled analysis"""
    analyzer = NewsAnalyzer()
    # Get API key from environment variable
    API_KEY = os.getenv('NEWS_API_KEY')
    if not API_KEY:
        print("Error: NEWS_API_KEY environment variable not set")
        return
    analyzer.run_analysis_pipeline(API_KEY)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart News Analyzer')
    parser.add_argument('--api-key', help='News API key (or set NEWS_API_KEY environment variable)')
    parser.add_argument('--mode', choices=['analyze', 'dashboard', 'schedule'], 
                       default='analyze', help='Run mode')
    parser.add_argument('--query', help='Search query for news')
    parser.add_argument('--sources', nargs='+', help='News sources to fetch from')
    
    args = parser.parse_args()
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.getenv('NEWS_API_KEY')
    
    if not api_key:
        print("Error: API key required. Either:")
        print("1. Set NEWS_API_KEY environment variable in .env file")
        print("2. Use --api-key argument")
        print("Get your free API key from: https://newsapi.org/")
        exit(1)
    
    if args.mode == 'analyze':
        analyzer = NewsAnalyzer()
        report = analyzer.run_analysis_pipeline(api_key, args.sources, args.query)
        print(json.dumps(report, indent=2))
    
    elif args.mode == 'dashboard':
        app.run(debug=True, host='0.0.0.0', port=5000)
    
    elif args.mode == 'schedule':
        # Schedule analysis to run every 6 hours
        schedule.every(6).hours.do(scheduled_analysis)
        
        print("Scheduled analysis started. Press Ctrl+C to stop.")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
