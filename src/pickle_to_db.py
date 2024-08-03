import pickle
import sqlite3

from settings import CN_DICT2_DB, CN_DICT2_P

with CN_DICT2_P.open("rb") as f:
    cn_dict = pickle.load(f)


conn = sqlite3.connect(CN_DICT2_DB)
c = conn.cursor()

c.execute("""CREATE TABLE cn_dict (lemma TEXT PRIMARY KEY, concepts TEXT)""")

for lemma, concepts in cn_dict.items():
    c.execute("INSERT INTO cn_dict (lemma, concepts) VALUES (?, ?)", (lemma, ",".join(concepts)))

conn.commit()
conn.close()
