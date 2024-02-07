
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import requests

list_df = pd.DataFrame(columns=['曲名','歌詞'])

#歌ネットのurlをbase_urlに入力
base_url = 'https://www.uta-net.com'
#urlに先ほど取得した歌詞一覧のURLを入力
url = 'https://www.uta-net.com/artist/684/4/'

#usr_agentに先ほど取得したUserAgent情報を入力
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

header = {'User-Agent': user_agent}

response = requests.get(url,headers=header)

soup = BeautifulSoup(response.text, 'lxml')

#引数として、class_='sp-w-100'を与える
links = soup.find_all('td', class_='sp-w-100')

#歌詞情報を取得
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

#曲名と歌詞が入っているlist_dfをcsvファイルに
df =list_df
df.to_csv('ファイル名.csv', index=False)
