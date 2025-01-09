# Finetune.py 

## Overview
This script fine-tunes a Vision Transformer (ViT) model for image classification tasks using parameter-efficient fine-tuning (PEFT). It uses stratified sampling for data splitting and provides functionalities for loading data, training, and evaluating the model. The script utilizes the user system's GPU for faster loading, training and evaluation. 

## Usage
- To use the model, follow instructions specified in README.md

## Limitations
1. `finetune.py` relies on the users GPU, thus, the size of the dataset is directly proportional to the training time. 
2. **Overfitting**: There is potential for overfitting as the number of epochs is defined by the user. 

## Vision
1. Implement methods to explore different combinations of hyperparameters for attaining efficient hyperparameter tuning. Include gradient accumulation methods to treat memory issues during training. 
2. Implement methods to prevent overfitting of the model.
3. Improve training time to aceelerate training on GPUs.
4. Include calculation of aditional evauation metrics like accuracy, precision, recall, F1-score.

Vision Transformers, is considered "black boxes" and lack interpretability. Understanding why the model makes certain predictions is a limitation as well as a potential for implementing a novel technique of measurement or monitoring the model's decisions.
