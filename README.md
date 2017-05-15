# kaggle-fisheries-monitoring

This repository contains code for the Nature Conservancy Fisheries Monitoring competition on Kaggle (https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring). In the end, it didn't do well enough to actually submit, but it was a good experience anyway. 

## Dependencies

This script was run with the dependencies listed below. I attempted to follow good programming practices, and so the code will probably run with updated versions of these libraries--but I make no guarantees of that. 
- Theano v0.8.2)
- Lasagne v0.1)
- Keras v2.0.2

## Run Instructions

1. First download the training data from the project website (https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data). 
2. Navigate to the top level of the project directory, and create a directory "data". Unzip the training data into "data". Modify the directory structure so that it is the following:
`.
+-- data
    |
    +-- train
    |   |
    |   +-- ALB
    |   |
    |   +-- BET
    |   |
    |   +-- DOL
    |   |
    |   +-- LAG
    |   |
    |   +-- NoF
    |   |
    |   +-- SHARK
    |   |
    |   +-- OTHER
    |   |
    |   +-- YFT
    |
    +-- validation
        |
        +-- ALB
        |
        +-- BET
        |
        +-- DOL
        |
        +-- LAG
        |
        +-- NoF
        |
        +-- SHARK
        |
        +-- OTHER
        |
        +-- YFT`
All of the training data downloaded should be in the training directory "data/train" and its subdirectories. The script automatically performs a random (stratified) sampling of the training set and puts those images into the validation directory (do not worry about this being problematic during multiple runs; the script re-combines all images into the train directory during every run before doing a train/validation split). 
3. Create a config file by copying the config_template.txt file into the a new file "config.txt". Then, modify the values in the config file as desired. (They are hyperparameters: initial learning rate, etc.; this is also where the train and validation directories are defined, so if you prefer different names or locations for those directories, you may change them here--but make sure that both the validation and train directories contain one subdirectory for each of the 8 classes). 
4. Run unit tests to make sure all the code is working properly on your system: `python utilities_test`. 
5. Run the script: `python train_and_test_network.py`. 

## Output

The output will be written to the log file "script_log.txt", and to standard output. The output includes the following:
* for each epoch:
    * the number of seconds the epoch took,
    * the average training loss over all batches in the epoch,
    * and the network's validation loss after the epoch is finished;
* and the gradient with respect to each batch during training.

