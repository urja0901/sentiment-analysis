from fastapi import FastAPI
from pydantic import BaseModel
import mlfoundry as mlf
import pandas as pd
import yaml
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI(docs_url="/")

def cleaning_stopwords(text, STOPWORDS):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def cleaning_punctuations(text, punctuations_list):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def cleaning_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def cleaning_email(data):
    return re.sub('@[^\s]+', ' ', data)

def cleaning_URLs(data):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',data)

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

def stemming_on_text(data, st):
    text = [st.stem(word) for word in data]
    return text

def lemmatizer_on_text(data, lm):
    text = [lm.lemmatize(word) for word in data]
    return text

@app.get("/")
async def root():
    return {"message": "Welcome to Twitter sentiment analysis inference"}

class SentimentAnalysis(BaseModel):
    tweet: str
 
with open("infer.yaml", "r") as stream:
    try:
        env_vars = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Load the model from MLFoundry by proving the MODEL_FQN
client = mlf.get_client(api_key=env_vars['components'][0]['env']['MLF_API_KEY'],tracking_uri=env_vars['components'][0]['env']['MLF_HOST'])
model_version = client.get_model(env_vars['components'][0]['env']['MODEL_FQN'])
model = model_version.load()

def preprocessed_tweet(tweet):
    data = pd.DataFrame()
    data['text'] = tweet

    STOPWORDS = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda text: cleaning_stopwords(text, STOPWORDS))
    english_punctuations = string.punctuation
    punctuations_list = english_punctuations

    data['text']= data['text'].apply(lambda x: cleaning_punctuations(x, punctuations_list))
    data['text'] = data['text'].apply(lambda x: cleaning_repeating_char(x))
    data['text']= data['text'].apply(lambda x: cleaning_email(x))
    data['text'] = data['text'].apply(lambda x: cleaning_URLs(x))
    data['text'] = data['text'].apply(lambda x: cleaning_numbers(x))

    #tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    data['text'] = data['text'].apply(tokenizer.tokenize)

    st = nltk.PorterStemmer()
    data['text']= data['text'].apply(lambda x: stemming_on_text(x, st))

    lm = nltk.WordNetLemmatizer()
    data['text'] = data['text'].apply(lambda x: lemmatizer_on_text(x, lm))

    X=data.text
    max_len = 500
    tok = Tokenizer(num_words=2000)
    tok.fit_on_texts(X)
    sequences = tok.texts_to_sequences(X)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

    return sequences_matrix

@app.post("/predict")
def predict(tweet):
    predict = [tweet]
    test = pd.DataFrame(data=[predict],columns=['tweet'])
    to_predict = preprocessed_tweet(test)

    print(to_predict.shape)
    prediction = model.predict(to_predict)
    prediction = (prediction > 0.5) 
    print(prediction)
    return {'sentiment' : prediction.tolist()[0]}

out = predict("It's a good day")
print(out)