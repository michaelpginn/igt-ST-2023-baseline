# Automatic-IGT-Glossing
This repository provides the baseline system for the task https://github.com/sigmorphon/2023glossingST. 

Train model:

```shell
python3 token_class_model.py train --lang ddo
```

Make predictions:

```shell
python3 token_class_model.py predict --lang ddo --pretrained_path ./output --data_path ../../GlossingSTPrivate/splits/Tsez/ddo-dev-track1-covered
```

Eval predictions:

```shell
python3 eval.py --pred ./predictions --gold ../../GlossingSTPrivate/splits/Tsez/ddo-train-track1-uncovered
```

## Model design

## Results

Trained models: [download here](https://o365coloradoedu-my.sharepoint.com/:f:/g/personal/migi8081_colorado_edu/EhzVMGQwS_5GuV4R1BZYbVIBJbj0zHi09t85zGRuAwEkbw?e=iEIIfH)

### Dev Performance
| Closed Track ||||||
| --- | --- | --- | --- | --- | --- |
| Lang | Morpheme Acc| Word Acc | BLEU (Morpheme) | Stems | Grams |
| --- | --- | --- | --- | --- | --- |
| ddo | Ovr: 47.5<br>Avg: 52.9 | Ovr: 71.8<br>Avg: 72.1 | 57.8 | P: 49.7<br>R: 49.2<br>F1: 49.4 | P: 50.7<br>R: 46.1<br>F1: 48.3 |
