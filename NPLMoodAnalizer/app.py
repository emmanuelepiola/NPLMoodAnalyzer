import os
import re
import csv
import pandas as pd
import emoji
from flask import Flask, render_template, jsonify
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# modify the path to the chat file
CHAT_FILE = "/Users/emmanuelepiola/documents/NPLMoodAnalizer/_chat.txt"
# modify the path to the csv file
CSV_FILE = "/Users/emmanuelepiola/documents/NPLMoodAnalizer/mood_scores.csv" 

# function to extract messages from the chat file
def extract_messages(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    participants = {}
    pattern = re.compile(r'\[\d{2}/\d{2}/\d{2}, \d{2}:\d{2}:\d{2}\] (.*?): (.*)')

    # emoji to text conversion
    for line in lines:
        match = pattern.match(line)
        if match:
            participant = match.group(1)
            msg = match.group(2)
            msg = emoji.demojize(msg)
            if participant not in participants:
                participants[participant] = []
            participants[participant].append(msg)

    return participants

# function to analyze the sentiment of the messages using the VADER NPL library
def analyze_sentiment(participants, output_csv):
    analyzer = SentimentIntensityAnalyzer()
    mood_trends = {participant: [] for participant in participants}
    message_scores = {participant: [] for participant in participants}

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Participant", "Message", "Sentiment", "Mood Score"])

        # translating from italian to english
        for participant, messages in participants.items():
            mood_score = 0
            for msg in messages:
                if msg.strip():
                    try:
                        translated_text = GoogleTranslator(source='auto', target='en').translate(msg)
                        sentiment = analyzer.polarity_scores(translated_text)['compound']
                    except:
                        sentiment = 0

                    # comulative mood score to track the mood trend with some context
                    mood_score += sentiment
                    mood_trends[participant].append(mood_score)
                    message_scores[participant].append((msg, sentiment))
                    writer.writerow([participant, msg, sentiment, mood_score])

    return mood_trends, message_scores

# simply displaying the data

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    if not os.path.exists(CSV_FILE):
        participants = extract_messages(CHAT_FILE)
        analyze_sentiment(participants, CSV_FILE)

    df = pd.read_csv(CSV_FILE)
    grouped = df.groupby("Participant")

    mood_data = {}
    for participant, group in grouped:
        mood_data[participant] = list(group["Mood Score"])

    best_worst_messages = {}
    for participant in mood_data.keys():
        subset = df[df["Participant"] == participant]
        max_msg = subset.loc[subset["Sentiment"].idxmax()]["Message"]
        min_msg = subset.loc[subset["Sentiment"].idxmin()]["Message"]
        best_worst_messages[participant] = {"max": max_msg, "min": min_msg}

    return jsonify({"mood_data": mood_data, "best_worst_messages": best_worst_messages})

if __name__ == "__main__":
    app.run(debug=True)
