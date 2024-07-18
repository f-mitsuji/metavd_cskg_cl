import pickle
import sqlite3

from settings import CN_DICT_P, CN_EN_LEMMAS_P

DB_PATH = "cn_dict2.db"
# DB_PATH = "conceptnet_lemmas.db"

# .pファイルの読み込み
with CN_DICT_P.open("rb") as f:
    conceptnet_lemmas = pickle.load(f)


# SQLiteデータベースの作成
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# テーブルの作成
c.execute("""CREATE TABLE lemmas
             (lemma TEXT PRIMARY KEY, uris TEXT)""")

# データの挿入
for lemma, uris in conceptnet_lemmas.items():
    c.execute("INSERT INTO lemmas (lemma, uris) VALUES (?, ?)", (lemma, ",".join(uris)))

# コミットして接続を閉じる
conn.commit()
conn.close()

print(f"Data from {CN_EN_LEMMAS_P} has been successfully converted to {DB_PATH}")
