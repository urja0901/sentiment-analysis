from operator import concat
import streamlit as st
import requests
import yaml
import base64


with open("demo.yaml", "r") as stream:
    try:
        env_vars = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

def main():
    request_url = env_vars['components'][0]['env']['INFER_URL']
    st.set_page_config(page_title="Sentiment Analysis")
    st.title("Sentiment Analysis Demo")
    st.header('Welcome to Sentiment Analysis inference!')
    st.write('This is a sample app that demonstrates the prowess of ServiceFoundry ML model deployment.🚀')
    st.write('Visit the [Github](https://github.com/urja0901/sentiment-analysis) repo for code or [Google Colab](https://colab.research.google.com/drive/1P6i8P9CvzeCa3iUDCb97-4YFWufCnh_S?usp=sharing) notebook for a quick start.')
    with st.form("my_form"):
        
        sentiment_text = st.text_input('Sentiment Text',value="It's a good day!")

        features = {
                "tweet": sentiment_text
            }
            
        submitted = st.form_submit_button("Submit")
        if submitted:
            data = requests.post(url=concat(request_url, "/predict"), params=features).json()
            if data:
                if data["sentiment"] == 0 :
                    return_val = "negative comment"
                else : 
                    return_val = "positive comment"
                print(return_val)
                st.metric(label="sentiment",value=return_val)
            else:
                st.error("Error")

    st.image('twitter2.jpeg', use_column_width='always')

if __name__ == '__main__':
    main()
