# MetaVD-CSKG Concept Linking

```text
root/
├── data/
│   ├── conceptnet/
│   │   ├── cn_dict2.p
│   │   └── concepts_en_lemmas.p
│   ├── embeddings/
│   │   ├── GoogleNews-vectors-negative300.bin
│   │   └── numberbatch-en-19.08.txt
│   ├── metavd/
│   │   ├── activitynet_classes.csv
│   │   ├── charades_classes.csv
│   │   ├── hmdb51_classes.csv
│   │   ├── kinetics700_classes.csv
│   │   ├── metavd_v1.csv
│   │   ├── stair_actions_classes.csv
│   │   └── ucf101_classes.csv
│   ├── models/
│   │   └── wiki_unigrams.bin
│   └── stop_words.txt
├── results/
│   └── ...
├── sent2vec/
│   └── ...
├── src/
│   └── ...
├── .gitignore
├── README.md
└── requirements.txt
```

## Data Files and Sources

### `data/conceptnet/`

- Contains two files:
  - `cn_dict2.p`
  - `concepts_en_lemmas.p`
- These files can be downloaded from the [CoCo-Ex GitHub repository](https://github.com/Heidelberg-NLP/CoCo-Ex).

### `data/embeddings/`

- Contains the following pre-trained embeddings:
  - [GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)
  - [numberbatch-en-19.08.txt](https://github.com/commonsense/conceptnet-numberbatch)

### `data/metavd/`

- Contains seven files:
  - `activitynet_classes.csv`
  - `charades_classes.csv`
  - `hmdb51_classes.csv`
  - `kinetics700_classes.csv`
  - `metavd_v1.csv`
  - `stair_actions_classes.csv`
  - `ucf101_classes.csv`
- These files are sourced from the [MetaVD GitHub repository](https://github.com/STAIR-Lab-CIT/metavd).
