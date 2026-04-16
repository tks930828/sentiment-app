import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# データ読み込み
df = pd.read_csv("data.csv")

@st.cache_resource
def train_model(df):
    y = df["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, vectorizer, X_test, y_test

model, vectorizer, X_test, y_test = train_model(df)

# ページ設定
st.set_page_config(page_title="感情分析", layout="centered")

# タイトル（中央寄せ）
st.markdown("<h1 style='text-align: center;'>感情分析アプリ</h1>", unsafe_allow_html=True)

# 余白
st.write("")

# 入力エリア（中央っぽくする）
col_center = st.columns([1, 2, 1])[1]

with col_center:
    text = st.text_input("文章を入力してください")
    analyze = st.button("分析")

# 実行
if analyze:
    input_vec = vectorizer.transform([text])
    prediction = model.predict(input_vec)
    proba = model.predict_proba(input_vec)[0]

    # カード風コンテナ
    st.markdown("---")

    result_col = st.columns([1, 2, 1])[1]

    with result_col:

        # 結果表示（中央）
        if prediction[0] == 1:
            st.markdown(
                f"<h2 style='text-align: center; color: green;'>😊 ポジティブ {proba[1]*100:.1f}%</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='text-align: center; color: red;'>😡 ネガティブ {proba[0]*100:.1f}%</h2>",
                unsafe_allow_html=True
            )

        # グラフ
        fig, ax = plt.subplots(figsize=(4, 2))

        labels = ["Positive", "Negative"]
        values = [proba[1], proba[0]]

        ax.barh(labels, values)
        ax.set_xlim(0, 1)

        for i, v in enumerate(values):
            ax.text(v, i, f"{v*100:.0f}%", va='center')

        plt.tight_layout()
        st.pyplot(fig)

        # 詳細（折りたたみ）
        with st.expander("詳細"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("ポジティブ", f"{proba[1]*100:.1f}%")
            with col_b:
                st.metric("ネガティブ", f"{proba[0]*100:.1f}%")