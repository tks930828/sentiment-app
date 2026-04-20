# sentiment-app
# 感情分析アプリ（Sentiment Analysis App）

## ■ 概要

テキストを入力すると、ポジティブまたはネガティブを判定するWebアプリです。
機械学習モデルを用いて文章の感情を自動分類します。

---

## ■ 作成背景

自然言語処理（NLP）に興味があり、テキストデータを扱う基本的な機械学習の流れを理解するために開発しました。
また、ユーザーが直感的に使えるUIを意識して実装しました。

---

## ■ 使用技術

* Python
* pandas
* scikit-learn

  * TfidfVectorizer
  * LogisticRegression
* Streamlit
* matplotlib

---

## ■ 機能

* テキスト入力による感情判定
* ポジティブ / ネガティブの確率表示
* グラフによる可視化
* モデル評価指標の表示（Accuracy, Precision, Recall, F1）

---

## ■ 技術的ポイント

### ・テキストの数値化

TF-IDFを用いてテキストを数値ベクトルに変換

### ・モデル選定

シンプルかつテキスト分類に適したLogisticRegressionを採用

### ・データ分割

train/testに分割し、過学習を防止

### ・UI設計

* 中央寄せレイアウト
* カード風デザイン
* 必要な情報のみ表示し、詳細は折りたたみ表示

---

## ■ 工夫した点

* predict_probaによる確率表示
* 横並びレイアウトによる視認性向上
* グラフによる直感的な理解
* キャッシュによる処理高速化

---

## ■ 今後の課題

* 日本語対応（形態素解析の導入）
* データセットの拡張による精度向上
* モデルの高度化（BERTなど）
* UI/UXのさらなる改善

---

## ■ 実行方法

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ■ デモ

（ここにスクリーンショットやURLを追加）

