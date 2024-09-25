# Audio-Text Retrieval in Context

This repository contains implementations of the paper Audio-Text Retrieval in Context, [arxiv](https://arxiv.org/abs/2203.13645). The implementation is partly based on [Audio Retrieval with Natural Language Queries](https://github.com/oncescuandreea/audio-retrieval) repo.

## Dataset

We use two datasets in this paper: [AudioCaps](https://aclanthology.org/N19-1011.pdf) and [CLOTHO](https://arxiv.org/pdf/1910.09387.pdf)

## Audio Feature Extraction

Six different feature extractor have been used in this study: log-mel spectrogram([librosa](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html)), [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), [VGGSound](https://github.com/hche11/VGGSound), [PANNs_CNN10](https://github.com/qiuqiangkong/audioset_tagging_cnn), [PANNs_CNN14](https://github.com/qiuqiangkong/audioset_tagging_cnn), [Efficient_Latent](https://github.com/RicherMans/HEAR2021_EfficientLatent).  

After extracting the audio features (preferred hdf5 file Format), the feature can be arrange following below structure under `./data`. 
```
.
└── data
    ├── AudioCaps
    │   ├── AudioExpert
    │   │   ├── panns_cnn14 
    │   │   │   ├── test
    │   │   │   │   └── panns_cnn14.h5
    │   │   │   ├── train
    │   │   │   │   └── panns_cnn14.h5
    │   │   │   └── val
    │   │   │       └── panns_cnn14.h5
    │   │   └── ...
    │   ├── Index
    │   │   ├── test.csv
    │   │   ├── train.csv
    │   │   ├── val.csv
    │   └── TextEmbeddings
    │       ├── w2v_test.pkl
    │       ├── w2v_train.pkl
    │       └── w2v_val.pkl
    └── CLOTHO
        ├── AudioExpert
        │   ├── panns_cnn14
        │   │   ├── test
        │   │   │   └── panns_cnn14.h5
        │   │   ├── train
        │   │   │   └── panns_cnn14.h5
        │   │   └── val
        │   │       └── panns_cnn14.h5
        │   └── ...
        ├── Index
        │   ├── test.json
        │   ├── train.json
        │   ├── val.json
        └── TextEmbeddings
            ├── w2v_test.pkl
            ├── w2v_train.pkl
            └── w2v_val.pkl
```

## Pooling Strategy

In this work, five pooling strategies were adopted: Mean Pooling, Max Pooling, LSTM, NetVLAD and NetRVLAD. For using different pooling strategies, can refer to different config files under `./configs/AudioCaps/` or `./cofigs/CLOTHO/`

## Reproduction
### AudioCaps
Training + evaluation:
```bash
python train.py --config configs/onfigs/AudioCaps/train-rvlad.json

```

### CLOTHO
Training + evaluation:
```bash
python train.py --config configs/onfigs/CLOTHO/train-rvlad.json

```



