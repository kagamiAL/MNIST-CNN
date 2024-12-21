# MNIST-CNN
MNIST-CNN is a convolutional neural network designed to train and infer on the MNIST database

## Usage
Make sure you have installed the dependencies listed in **requirements.txt**

The following instructions assume you are in the MNIST-CNN directory

### Running

MNIST-CNN comes with a PyGame application where the model
tries guessing the number you've drawn

You can choose which model the application will use by
changing the "name" field in **parameters.json**. Models must be a **.pt** file
```json
//Do not specify the path just the name
//Assumes your model is in the out/ directory
"name": "THE NAME OF YOUR MODEL.pt"
```
To start the application simply run the command:
```bash
python recognition.py
```

### Playing

Hold down your left mouse button to draw on the canvas

Pressing backspace will clear the entire canvas

Pressing enter will let the model guess what number you've drawn
- The guess will be shown in your terminal

### Training
MNIST-CNN was designed to be trained on the data provided by: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/

The data must be a .csv file and follow this format:
- The first column is the label (a number from 0 to 9)
- The remaining 784 columns are the pixel values (0 to 255) for the 28 x 28 image

Once you have data that is formatted correctly, make sure
that in **parameters.json** the following fields point to the right paths
```json
"train_csv": "PATH_TO_TRAINING_DATA_CSV",
"val_csv": "PATH_TO_VALIDATION_DATA_CSV"
```

You are ready to train! Feel free to tweak the parameters to test out model performance

To start training simply run the command:
```bash
python digit_recognition_trainer.py
```

Models will be saved in the out/ directory
## Authors
Alan Bach, bachalan330@gmail.com
