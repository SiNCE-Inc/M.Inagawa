# -*- coding: utf-8 -*-
"""0.可視化用スクレイピング.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Hk9Hc-igAYsVVhBC0eOs2Myuc9M4_L2C
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import requests

list_df = pd.DataFrame(columns=['曲名','歌詞'])

#歌ネットのurlをbase_urlに入力します
base_url = 'https://www.uta-net.com'
#urlに先ほど取得した歌詞一覧のURLを入力します
url = 'https://www.uta-net.com/artist/684/4/'

#usr_agentに先ほど取得したUserAgent情報を入力します
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

header = {'User-Agent': user_agent}

response = requests.get(url,headers=header)

soup = BeautifulSoup(response.text, 'lxml')

#引数として、class_='sp-w-100'を与えます
links = soup.find_all('td', class_='sp-w-100')

#歌詞情報を取得します
for link in links:
    a = base_url + (link.a.get('href'))
    response = requests.get(a)
    soup = BeautifulSoup(response.text, 'lxml')
    song_name = soup.find('h2').text

    song_kashi = soup.find('div', id="kashi_area")
    song_kashi = song_kashi.text

    time.sleep(1)

    tmp_se = pd.DataFrame([[song_name], [song_kashi]],index=list_df.columns).T

    list_df = list_df.append(tmp_se)

list_df.head()

df =list_df

df.to_csv('Mr.Children.csv', index=False)

!pip install japanize_matplotlib

!pip install fugashi

!pip install ipadic

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import japanize_matplotlib
# %matplotlib inline

from transformers import BertJapaneseTokenizer, BertModel

# BERTの日本語モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

#トークナイザとモデルのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model = model.cuda()

#各データの形式を整える
max_length = 256

sentence_vectors = []
labels = []
for i in range(len(df)):
    # 記事から文章を抜き出し符号化を行う
    lines = df.iloc[i,3].splitlines()
    text = '\n'.join(lines)
    encoding = tokenizer(
        text,
        max_length = max_length,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt'
        )
    encoding = {k: v.cuda() for k, v in encoding.items()}
    attention_mask = encoding['attention_mask']

    #文章ベクトルを計算
    with torch.no_grad():
        output = model(**encoding)
        last_hidden_state = output.last_hidden_state
        averaged_hidden_state =(last_hidden_state*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(1,keepdim=True)

    #文章ベクトルとラベルを追加
    sentence_vectors.append(averaged_hidden_state[0].cpu().numpy())
    label = df.iloc[i,4]
    labels.append(label)

#ベクトルとラベルをnumpy.ndarrayにする
sentence_vectors = np.vstack(sentence_vectors)
labels = np.array(labels)

import torch

print(torch.cuda.is_available())

df = df.iloc[1:10]

from transformers import BertJapaneseTokenizer, BertModel
import torch
import numpy as np

# BERTの日本語モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# トークナイザとモデルのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# 各データの形式を整える
max_length = 256

sentence_vectors = []

for i in range(len(df)):
    # 記事から文章を抜き出し符号化を行う
    lines = df.iloc[i, 1].splitlines()
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

# Pythonリストとして取得
sentence_vectors = np.vstack(sentence_vectors)

perplexity_value = min(30, 1)  # 30は適当な上限値
sentence_vectors_tsne = TSNE(n_components=2, perplexity=perplexity_value).fit_transform(sentence_vectors)

import matplotlib.pyplot as plt

# t-SNEによる次元削減結果を散布図として可視化
plt.scatter(sentence_vectors_tsne[:, 0], sentence_vectors_tsne[:, 1])
plt.title('t-SNE Visualization of Sentence Vectors')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()