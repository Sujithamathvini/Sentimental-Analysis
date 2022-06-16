from flask import Flask, render_template,request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.downloader.download('vader_lexicon')

app = Flask(__name__)

@app.route('/result',methods=["GET","POST"])
def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(inp)
        pos = score["pos"]
        neg = score["neg"]
        neu = score["neu"]
        maxi = max(pos,neg,neu)
        if maxi == score["neg"]:
            return render_template('negative.html', msg="negative")
        elif maxi == score["pos"]:
            return render_template('positive.html',msg="Positive")
        else:
            return render_template('neutral.html', msg="Neutral")
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    return render_template('home.html')





