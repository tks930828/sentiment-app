import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# データ読み込み
df = pd.read_csv("data.csv")
y = df["label"]

# データ分割
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"], y, test_size=0.2, random_state=42
)

# ベクトル化
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# モデル学習
model = LogisticRegression()
model.fit(X_train, y_train)

# 評価
y_pred = model.predict(X_test)

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1:", f1_score(y_test, y_pred))

# UI
st.title("感情分析アプリ")

text = st.text_input("文章を入力してください")

if st.button("分析"):
    input_vec = vectorizer.transform([text])
    prediction = model.predict(input_vec)

    if prediction[0] == 1:
        st.write("😊 ポジティブ")
    else:
        st.write("😡 ネガティブ")