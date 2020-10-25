# Kannada-MNIST

Code to train a convolutional neural network to classify the digits of the Kannada language.

Files structure:
- main.py: The main function to run for training and testing the neural network.
- dataset.py: Contains functions and dataset class for handling input data.
- convnet.py: Defines the convolutional neural network architecture.
- training.py: Code for training the neural network and saving trained model and plots.
- testing.py: Runs the model and generates output file for submission.
- unittests.py: Performs unittest for the neural network model.

## Dataset

To run the code, the Kannada-MNIST dataset needs to be downloaded from:
https://www.kaggle.com/c/Kannada-MNIST/data

The data has to be placed within the same directory as the python files to work with the default parameters. Otherwise path to the dataset files has to be provided through the command-line.


## Run unittest

Command to run the unittests:

```
python3 unittests.py
```

## Reproducting results

To train the neural network and output the submission file, run:
```
python3 main.py
```

If desired, the main.py file contains options for command-line arguments which are explained with argumentparser. It enables the input/output files to be defined in the command-line and further it enables selection between training, testing or both.

The default arguments reproduce the obtained results.
