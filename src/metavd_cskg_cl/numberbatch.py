from gensim.models import KeyedVectors

model_path = "numberbatch-en-19.08.txt"

model = KeyedVectors.load_word2vec_format(model_path, binary=False)

similar_words = model.most_similar("playing_soccer", topn=10)
for word, similarity in similar_words:
    print(f"{word}: {similarity}")
