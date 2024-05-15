import numpy as np
from gensim.models import KeyedVectors

model_path = "numberbatch-en-19.08.txt"
model = KeyedVectors.load_word2vec_format(model_path, binary=False)


def find_similar_entities(action_label, topn=10):
    words = action_label.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return "None of the words in the label are in the Numberbatch vocabulary."

    average_vector = np.mean(word_vectors, axis=0)

    return model.similar_by_vector(average_vector, topn=topn)


action_label = "playing soccer"
similar_entities = find_similar_entities(action_label)
print(similar_entities)
