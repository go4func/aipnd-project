# AI Programming with Python Project

## Resource

- data folder: default to `flowers`, contains train, validation and test data(flower images).
- out folder: default to `./`, contains checkpoints as result from train script.
- cat_to_name.json: mapping from category to name.
- Makefile: for linux, contains scripts to run training and prediction.

## Part 1: Development Notebook

Notebook: [Image Classifier Project.ipynb](./Image%20Classifier%20Project.ipynb)

note: the checkpoint of this notebook can not be used for train script of part 2, since it has fixed pre-trained model(arch): `vgg16_bn`.

## Part 2: Command Line Application

### Scripts

- [fc_classifier.py](./fc_classifier.py): contains an implementation of feed-forward classifier with custom network args: input_size, output_size, hidden_layer, drop_p.
- [util.py](./util.py): contains functions that used in train and predict script.
- [train.py](./train.py): main script to run training for network.
- [predict.py](./predict.py): main script to run image prediction.

### train.py

```
usage: train.py [-h] [--data_dir DATA_DIR] [--checkpoint CHECKPOINT] [--arch {vgg16_bn,densenet121}] [--learning_rate LEARNING_RATE]
                        [--hidden_units HIDDEN_UNITS] [--gpu] [--epochs EPOCHS]

Train the network and use it to classify the flowers

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   location of train, valid and test data
  --checkpoint CHECKPOINT
                        location for saving trained network checkpoint
  --arch {vgg16_bn,densenet121}
                        choose architecture
  --learning_rate LEARNING_RATE
                        learning rate of network
  --hidden_units HIDDEN_UNITS
                        units of hidden layer
  --gpu                 user gpu to train network
  --epochs EPOCHS       training epochs

```

Example:

```
python train.py --data_dir=flowers --checkpoint='out/vgg16_bn_checkpoint.pth' --arch=vgg16_bn --learning_rate=0.001 --hidden_units=1028 --gpu --epochs=7
```

### predict.py

```
usage: predict.py [-h] [--image IMAGE] [--checkpoint CHECKPOINT] [--class_map CLASS_MAP] [--topk TOPK] [--gpu]

Predict the image

options:
  -h, --help            show this help message and exit
  --image IMAGE         path to the image to be predicted
  --checkpoint CHECKPOINT
                        the model checkpoint file, result from training script.
  --class_map CLASS_MAP
                        json file that has the map of classes to flower names.
  --topk TOPK           top k classes.
  --gpu                 user gpu to predict image
```

Example:

```
python predict.py --image='flowers/test/11/image_03130.jpg' --checkpoint='out/vgg16_bn_checkpoint.pth' --class_map=cat_to_name.json --topk=5 --gpu
```
