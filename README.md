# Crop Disease Classification in Uganda

Cassava leaf classification using a compact CNN and ResNet50 transfer learning,
completed in the context of WorldQuant University's Applied AI Lab.

## Data

The five classes are cassava bacterial blight, brown streak disease, green mottle,
mosaic disease, and healthy leaves. Use authorized course images in one folder
per class, or the [Kaggle cassava dataset](https://www.kaggle.com/competitions/cassava-leaf-disease-classification).
The course also references the [merged 2019/2020 dataset](https://www.kaggle.com/datasets/srg9000/cassava-plant-disease-merged-20192020).

For the official `train.csv` and `train_images/` layout:

```bash
python data.py --images data/train_images --labels data/train.csv --output data/train
```

Course folders can be used directly, for example `data_p2/data_undersampled/train`.
The folder names expected by the scripts are listed in `data.py`.

## Setup and usage

Use Python 3.11 and a separate virtual environment for this project.

```bash
python -m pip install -r requirements.txt
python train.py --data data/train
python predict.py --checkpoint outputs/resnet50/model.pt --image data/example.jpg
```

ResNet50 uses ImageNet weights and trains its new classifier by default. Add
`--fine-tune` to update the backbone. To train the compact CNN without downloading
pretrained weights:

```bash
python train.py --data data/train --model cnn --output outputs/cnn
```

The pipeline uses a seeded, stratified 80/20 split, training-only augmentation,
ImageNet normalization, Adam, weight decay, StepLR scheduling, and early stopping.
The frozen ResNet backbone retains its pretrained batch-normalization statistics.
Training saves the minimum-validation-loss checkpoint, history, and confusion
matrix. Prediction returns a class and five probabilities.

## Evaluation

Report validation accuracy together with per-class confusion counts because the
original dataset is imbalanced. No new cassava accuracy is claimed without a
training run on the authorized dataset. Splits and preprocessing in these scripts
can differ from the course notebooks.

## References and license

[WorldQuant University](https://www.wqu.edu/) provided the course and original
notebooks. [torchvision ResNet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
provides the pretrained backbone. The standalone scripts and documentation use
MIT; original course materials and datasets retain their own terms.
