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
