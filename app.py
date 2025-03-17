import os
from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from prophet import Prophet
from pytrends.request import TrendReq
import numpy as np 
import plotly.graph_objects as go
from datetime import datetime

# Fix for matplotlib backend issues in Flask
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Error reading file: {str(e)}", 400

    if 'review' not in df.columns:
        return "CSV must contain a 'review' column", 400

    df['review'] = df['review'].fillna("")  
    df['sentiment'] = df['review'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)

    # Save histogram
    plt.figure(figsize=(10, 5))
    plt.hist(df['sentiment'], bins=20, color='blue', edgecolor='black')
    plt.xlabel("Sentiment Score")
    plt.ylabel("Count")
    plt.title("Sentiment Analysis Distribution")
    sentiment_hist_path = 'static/sentiment_analysis.png'
    plt.savefig(sentiment_hist_path)
    plt.close()

    # Categorize sentiments
    positive = sum(df['sentiment'] > 0)
    neutral = sum(df['sentiment'] == 0)
    negative = sum(df['sentiment'] < 0)

    # Save pie chart
    plt.figure(figsize=(6, 6))
    plt.pie([positive, neutral, negative], labels=['Positive', 'Neutral', 'Negative'], autopct='%1.1f%%', colors=['green', 'gray', 'red'])
    plt.title("Sentiment Breakdown")
    sentiment_pie_path = 'static/sentiment_pie_chart.png'
    plt.savefig(sentiment_pie_path)
    plt.close()

    return render_template("sentiment_result.html", 
                           image_url="/static/sentiment_analysis.png", 
                           pie_chart_url="/static/sentiment_pie_chart.png")

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Error reading file: {str(e)}", 400

    if 'Date' not in df.columns or 'Sales' not in df.columns:
        return "CSV must contain 'Date' and 'Sales' columns", 400

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Save forecast plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['ds'], df['y'], label='Actual Sales', marker='o')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Sales', linestyle="dashed", color="red")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Sales Prediction")
    plt.legend()
    plt.savefig('static/sales_prediction.png')
    plt.close()

    return render_template("sales_result.html", image_url="/static/sales_prediction.png")


@app.route('/market_research', methods=['POST'])
def market_research():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Error reading file: {str(e)}", 400

    if 'keyword' not in df.columns:
        return "CSV must contain a 'keyword' column", 400

    keyword = df['keyword'][0]  # Take the first keyword for analysis

    try:
        pytrends = TrendReq()
        
        # Define timeframes for different trends
        timeframes = {
            "daily": "now 7-d",
            "monthly": "today 3-m",
            "yearly": "today 12-m"
        }
        
        image_paths = {}

        for period, timeframe in timeframes.items():
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='US', gprop='')
            df_trend = pytrends.interest_over_time()

            if df_trend.empty:
                image_paths[f"{period}_image_url"] = None
                continue

            # Plot and save the graph
            plt.figure(figsize=(10, 5))
            plt.plot(df_trend.index, df_trend[keyword], marker="o", linestyle="dashed", color="blue", label=keyword)
            plt.xlabel("Date")
            plt.ylabel("Interest Level")
            plt.title(f"{period.capitalize()} Trend Analysis for '{keyword}'")
            plt.legend()

            image_path = f'static/{period}_trend.png'
            plt.savefig(image_path)
            plt.close()

            image_paths[f"{period}_image_url"] = f"/{image_path}"

        return render_template("market_research_result.html", 
                               daily_image_url=image_paths["daily_image_url"],
                               monthly_image_url=image_paths["monthly_image_url"],
                               yearly_image_url=image_paths["yearly_image_url"])

    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Error reading file: {str(e)}", 400

    if 'Date' not in df.columns or 'Close' not in df.columns:
        return "CSV must contain 'Date' and 'Close' columns", 400

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Save forecast plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['ds'], df['y'], label='Actual Close Price', marker='o')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Close Price', linestyle="dashed", color="red")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Stock Market Prediction")
    plt.legend()
    plt.savefig('static/stock_prediction.png')
    plt.close()

    # Generate analysis
    predicted_trend = "increasing" if forecast['yhat'].iloc[-1] > df['y'].iloc[-1] else "decreasing"
    predicted_high = round(forecast['yhat_upper'].max(), 2)
    predicted_low = round(forecast['yhat_lower'].min(), 2)

    analysis_text = f"""
        Based on AI predictions, the stock trend is expected to be **{predicted_trend}** in the next 30 days. 
        The highest predicted price could reach **₹{predicted_high}**, while the lowest could drop to **₹{predicted_low}**. 
        Please note that market conditions, news, and external factors can influence actual stock performance.
    """

    return render_template("stock_market_result.html", 
                           image_url="/static/stock_prediction.png", 
                           analysis_text=analysis_text)

@app.route('/report_issue', methods=['POST'])
def report_issue():
    issue_text = request.form.get('issue')
    
    if not issue_text:
        return "No issue provided", 400

    # Save the issue to a file (optional)
    with open("reported_issues.txt", "a") as file:
        file.write(f"{datetime.now()} - {issue_text}\n")

    # Print the issue in the terminal
    print(f"[Issue Reported] {datetime.now()} - {issue_text}")

    return render_template("reported_issues_result.html")

if __name__ == '__main__':
    app.run(debug=True)
