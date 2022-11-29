import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import re
from nltk.stem import PorterStemmer 
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

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



def get_initial_data(data_path):
    data = pd.read_csv(data_path, encoding = "ISO-8859-1")
    data = data.sample(1000)
    data.columns = ["label", "time", "date", "query", "username", "text"]
    data=data[['text','label']]
    data['label'][data['label']==4]=1
    data_pos = data[data['label'] == 1]
    data_neg = data[data['label'] == 0]
    data_pos = data_pos.iloc[:int(20000)]
    data_neg = data_neg.iloc[:int(20000)]

    #combine data
    data = pd.concat([data_pos, data_neg])
    data['text']=data['text'].str.lower()

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
    y=data.label
    max_len = 500
    tok = Tokenizer(num_words=2000)
    tok.fit_on_texts(X)
    sequences = tok.texts_to_sequences(X)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    X_train, X_test, Y_train, Y_test = train_test_split(sequences_matrix, y, test_size=0.3, random_state=2)
    
    return X_train, X_test, Y_train, Y_test