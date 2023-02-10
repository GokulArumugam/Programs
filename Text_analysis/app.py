from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

app = Flask(__name__)

def load_data(file):
    df = pd.read_excel(file)
    return df

def generate_wordcloud(df):
    all_words = ' '.join([text for text in df['feedback']])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

def generate_sentiment_table(df):
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    sentiments = []
    for index, row in df.iterrows():
        score = sid.polarity_scores(row['feedback'])
        sentiments.append(score)
    sentiments_df = pd.DataFrame(sentiments)
    df = pd.concat([df, sentiments_df], axis=1)
    positive_phrases = df[df['pos'] == 1].drop(['neg', 'neu', 'compound'], axis=1)
    negative_phrases = df[df['neg'] == 1].drop(['pos', 'neu', 'compound'], axis=1)
    return positive_phrases, negative_phrases

def sentiment_analysis(df):
    sentiment = []
    for index, row in df.iterrows():
        analysis = TextBlob(row['feedback'])
        sentiment.append(analysis.sentiment.polarity)
    df['sentiment'] = sentiment
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = load_data(file)
            generate_wordcloud(df)
            positive_phrases, negative_phrases = generate_sentiment_table(df)
            df = sentiment_analysis(df)
            return render_template('index.html', tables=[positive_phrases.to_html(classes='positive'), negative_phrases.to_html(classes='negative')])
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
