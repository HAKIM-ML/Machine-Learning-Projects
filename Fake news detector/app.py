import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]

    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)
tfdf = pickle.load(open('F:\\class\\Machine Learning\\project\\Fake news detector\\vectorizer.pkl','rb'),encoding='utf-8')
model = pickle.load(open('F:\\class\\Machine Learning\\project\\Fake news detector\\model.pkl','rb'),encoding='utf-8')

st.title("Message or mail Spam or Ham detector")

input_sms = st.text_area("Enter Message: ")
if st.button("predict"):
    transform_sms = transform_text(input_sms)
    vettot_input = tfdf.transform([transform_sms])
    result = model.predict(vettot_input)[0]
    if result == 1:
        st.header("Not Spam")
    else:
        st.header("spam")