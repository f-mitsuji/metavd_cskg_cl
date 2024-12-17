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
│   └── stop_words.txt
├── models/
│   └── ...
├── results/
│   └── ...
├── sent2vec/
│   ├── models/
│   │   └── wiki_unigrams.bin
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

### `models/trained/`

- Contains model weights:
  - `r2plus1d_activitynet.pth`: R(2+1)D model fine-tuned for ActivityNet action recognition
  - `r2plus1d_charades.pth`: R(2+1)D model fine-tuned for Charades action recognition
  - `r2plus1d_hmdb51.pth`: R(2+1)D model fine-tuned for HMDB51 action recognition
  - `r2plus1d_stair_actions.pth`: R(2+1)D model fine-tuned for STAIR Actions action recognition
  - `r2plus1d_ucf101.pth`: R(2+1)D model fine-tuned for UCF101 action recognition
  <!-- - `trained/`: Fine-tuned models
    - `hmdb/`: Models fine-tuned on HMDB51
      - `r2plus1d_hmdb.pth`: R(2+1)D model fine-tuned for HMDB51 action recognition
    - `ucf101/`: Models fine-tuned on UCF101
      - `r2plus1d_ucf101.pth`: R(2+1)D model fine-tuned for UCF101 action recognition -->

For fine-tuned models, please download from [here](https://drive.google.com/drive/folders/16NvVAkQhy7VYdggQZAu85R8FUXjXEQJC?usp=drive_link).
