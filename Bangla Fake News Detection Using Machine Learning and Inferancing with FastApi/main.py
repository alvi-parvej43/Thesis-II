#importing libraries for data cleaning
import nltk
from bs4 import BeautifulSoup
import re,string,unicodedata
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI
import sklearn


#initilized fast api
app = FastAPI(
    title="News Model API",
    description="A simple API that use NLP model to predict the news if it is fake or real",
    version="0.1",
)

#load model
with open(
    join(dirname(realpath(__file__)), "news_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)


#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


#remove special character
def rem_special_characters(text):
    pattern = re.compile('[!@#$%^&*()_+-={}\[\];:\'\"\|<>,.///?`~।]', flags=re.I)
    return pattern.sub(r'', text)

#remove non bangla character
def rem_non_bangla_characters(text):
    pattern = re.compile('[A-Z]', flags=re.I)
    return pattern.sub(r'', text)

#remove non empticons character
def rem_emoticons(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


#fuction for tokenizing text
def tokenized_data(sent):
    tokenized_text = sent.split()
    return tokenized_text

stop_words = { "অবশ্য" ,"অনেক","অনেকে","অনেকেই","অন্তত","অথবা","অথচ","অর্থাত","অন্য","আজ","আছে","আপনার","আপনি","আবার","আমরা"
"আমাকে","আমাদের","আমার","আমি","আরও","আর","আগে","আগেই","আই","অতএব","আগামী","অবধি","অনুযায়ী","আদ্যভাগে","এই","একই","একে",
"একটি","এখন","এখনও","এখানে","এখানেই","এটি","এটা","এটাই","এতটাই","এবং","একবার","এবার","এদের","এঁদের","এমন","এমনকী","এল",
"এর","এরা","এঁরা","এস","এত","এতে","এসে","একে","এ","ঐ","ই","ইহা","ইত্যাদি","উনি","উপর","উপরে","উচিত","ও","ওই","ওর","ওরা","ওঁর","ওঁরা",
"ওকে","ওদের","ওঁদের","ওখানে",'কত',"কবে",'করতে',"কয়েক""কয়েকটি","করবে","করলেন","করার","কারও","করা","করি","করিয়ে","করার",
"করাই","করলে","করলেন","করিতে","করিয়া","করেছিলেন","করছে","করছেন","করেছেন","করেছে","করেন","করবেন","করায়","করে","করেই",
"কাছ","কাছে","কাজে","কারণ","কিছু","কিছুই","কিন্তু","কিংবা","কি","কী","কেউ","কেউই","কাউকে","কেন","কে","কোনও","কোনো","কোন",
"কখনও","ক্ষেত্রে","খুব","গুলি","গিয়ে","গিয়েছে","গেছে","গেল","গেলে","গোটা","চলে","ছাড়া","ছাড়াও","ছিলেন","ছিল","জন্য","জানা","ঠিক","তিনি",
"তিনঐ","তিনিও","তখন","তবে","তবু","তাঁদের","তাঁাহারা","তাঁরা","তাঁর","তাঁকে","তাই","তেমন","তাকে","তাহা","তাহাতে","তাহার","তাদের","তারপর",
"তারা","তারৈ","তার","তাহলে","তিনি","তা","তাও","তাতে","তো","তত","তুমি","তোমার","তথা","থাকে","থাকা","থাকায়","থেকে","থেকেও","থাকবে",
"থাকেন","থাকবেন","থেকেই","দিকে","দিতে","দিয়ে","দিয়েছে","দিয়েছেন","দিলেন","দু","দুটি","দুটো","দেয়","দেওয়া","দেওয়ার","দেখা","দেখে","দেখতে",
"দ্বারা","ধরে","ধরা","নয়","নানা","না","নাকি","নাগাদ","নিতে","নিজে","নিজেই","নিজের","নিজেদের","নিয়ে","নেওয়া","নেওয়ার","নেই","নাই","পক্ষে",
"পর্যন্ত","পাওয়া","পারেন","পারি","পারে","পরে","পরেই","পরেও","পর","পেয়ে","প্রতি","প্রভৃতি","প্রায়","ফের","ফলে","ফিরে","ব্যবহার","বলতে",
"বললেন","বলেছেন","বলল","বলা","বলেন","বলে","বহু","বসে","বার","বা","বিনা","বরং","বদলে","বাদে","বার","বিশেষ","বিভিন্ন","বিষয়টি","ব্যবহার","ব্যাপারে""ভাবে","ভাবেই","মধ্যে","মধ্যেই",
"মধ্যেও","মধ্যভাগে","মাধ্যমে","মাত্র","মতো","মতোই","মোটেই","যখন","যদি","যদিও","যাবে","যায়","যাকে","যাওয়া","যাওয়ার","যত","যতটা","যা","যার",
"যারা","যাঁর","যাঁরা","যাদের","যান","যাচ্ছে","যেতে","যাতে","যেন","যেমন","যেখানে","যিনি","যে","রেখে","রাখা","রয়েছে","রকম","শুধু","সঙ্গে",
"সঙ্গেও","সমস্ত","সব","সবার","সহ","সুতরাং","সহিত","সেই","সেটা","সেটি","সেটাই","সেটাও","সম্প্রতি","সেখান","সেখানে","সে","স্পষ্ট","স্বয়ং","হইতে",
"হইবে","হৈলে","হইয়া","হচ্ছে","হত","হতে","হতেই","হবে","হবেন","হয়েছিল","হয়েছে","হয়েছেন","হয়ে","হয়নি","হয়","হয়েই","হয়তো","হল","হলে",
"হলেই","হলেও","হলো","হিসাবে","হওয়া","হওয়ার","হওয়ায়","হন","হোক","জন","জনকে","জনের","জানতে","জানায়","জানিয়ে","জানানো","জানিয়েছে",
"জন্য","জন্যওজে","জে","বেশ","দেন","তুলে","ছিলেন","চান","চায়","চেয়ে","মোট","যথেষ্ট","টি"}

def remove_stop_words(text):
    tokens = tokenized_data(text)
    text = [w for w in tokens if not w in stop_words]
    text = ' '.join(text)
    return text

# cleaning the data
def text_cleaning(text):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = rem_special_characters(text)
    text = rem_non_bangla_characters(text)
    text = rem_emoticons(text)
    text = remove_stop_words(text)

    return text

@app.get("/predict-news")

@app.get("/predict-news")
def predict_news(news: str):
    """
    A simple function that receive a news content content and predict if the news is real or fake.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = text_cleaning(news)
    cleaned_review = remove_between_square_brackets(news)

    # vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)

    
    # perform prediction
    prediction = model.predict([cleaned_review])
    output = int(prediction[0])
    probas = model.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    Content = {0: "Fake", 1: "Real"}
    
    # show results
    result = {"prediction": Content[output], "Probability": output_probability}
    return result

#uvicorn main:app --reload (reload api)

# http://127.0.0.1:8000/docs (url link)