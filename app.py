import pandas as pd
import streamlit as st

# データ読み込み
df = pd.read_csv("data.csv")

# 特徴量変換
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# モデル学習
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, df["label"])

# UI
st.title("感情分析アプリ")

text = st.text_input("文章を入力してください")

if st.button("分析"):
    X_test = vectorizer.transform([text])
    prediction = model.predict(X_test)

    if prediction[0] == 1:
        st.write("😊 ポジティブ")
    else:
        st.write("😡 ネガティブ")