#必要に応じてダウンロード
!pip install japanize_matplotlib
!pip install fugashi
!pip install ipadic
!pip install transformers
!pip install torch
!pip install torch transformers --upgrade

#使うライブラリを呼び出します
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import torch
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#ファイルを読み込み
df= pd.read_csv(csv_path)

# BERTの日本語モデルをダウンロード
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# トークナイザとモデルのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# 各データの形式を整える
max_length = 256

###文字列をベクトル化
sentence_vectors = []
for i in range(len(df)):
    # 記事から文章を抜き出し符号化を行う
    lines = df.iloc[i, 0].splitlines()
    text = '\n'.join(lines)
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 文章ベクトルを計算
    with torch.no_grad():
        output = model(**encoding)
        last_hidden_state = output.last_hidden_state
        attention_mask = encoding['attention_mask']
        averaged_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

    # 計算された文章ベクトルを追加
    sentence_vectors.append(averaged_hidden_state.numpy())

# リストの形式を変更
sentence_vectors = np.vstack(sentence_vectors)


###k平均法によるクラスタリング
#(1)エルボー法により、クラスター数を決める
#クラスター数を[1,10]の中から選びたい場合
cluster_range = range(1, 10)
inertia_values = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=22)
    kmeans.fit(sentence_vectors)
    inertia_values.append(kmeans.inertia_)
    
# エルボー曲線のプロット
plt.plot(cluster_range, inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.title('Elbow Method for Optimal Cluster Number')
plt.show()

#(2)モデルを作成
kmeans = KMeans(n_clusters=10, random_state=22)
kmeans.fit(sentence_vectors)
