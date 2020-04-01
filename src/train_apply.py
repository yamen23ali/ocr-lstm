import sys

import os
import torch
import numpy as np
import torch.optim as optim
import time

from ocr_data_loader import load_data
from ocr_image_transformations import *
from ocr_model import OCRModel
from torchvision import transforms



def train_apply(model_name, base_dir= 'GT4HistOCR/corpus', dataset_name= 'RefCorpus-ENHG-Incunabula', holdout_book = '1499-CronicaCoellen',
	frame_size=1, max_image_height=20, max_image_width=500, hidden_layer_size=100, hidden_layer_num=3, 
	learning_rate=0.01, momentum=0.9, epochs=2000, clipping_value=1000, batch_size=300, dropout_ratio=0.2):
    
  """
  Load the data, split it , train the model and return the predictions on
  ( Train, Test, Holdout) data sets.

  Args:
    model_name (str): Unique name for the model to save it under it
    base_dir:  (str): The base directory of the GT4Hist data
    dataset_name (str): The name of the data set to train on
    holdout_book (str): The name of the book to keep as holdout (i.e. never train on it)
    frame_size (int): The size of the frame (i.e vertical cut) we would take from the text line image
    max_image_height (int): The maximum height we allow for a text line image (i.e all image will be scaled accordingly using Thumbnail transformation)
    max_image_width (int): The maximum width we allow for a text line image (i.e all image will be scaled accordingly using Thumbnail transformation)
    hidden_layer_size (int): The number of hidden units in the BLSTM cell
    hidden_layer_num (int): The number of BLSTM cells
    learning_rate (float): The learning rate of the SGD algorithm
    momentum (float): The momentum of the SGD algorithm
    epohcs (int): The number of iterations in the SGD algorithm
    clipping_value (float): The value to clip the grdient at
    batch_szie (int): How many samples to train on in each forward pass
    dropout_ratio (float): The probablity of dropping out a neuron during the learning process

  Returns:
    (:obj: `dict`, :obj: `dict`, :obj: `dict`): The results of predctions on ( Train, Test, Holdout) data after the model is trained
  """
  
  TRAIN_TEST_SPLIT = .8

  # ============= Load and prepare Data
  transformation = transforms.Compose([
    FixLineImage(),
    ImageThumbnail(max_image_height, max_image_width),
    transforms.ToTensor(),
    ImageTensorPadding(max_image_height, max_image_width),
    UnfoldImage(1, frame_size)
  ])

  train_data, test_data, holdout_data, dataset = load_data(base_dir = base_dir, dataset_name = dataset_name,
                                           holdout_book=holdout_book, transformation=transformation,
                                           train_test_split=TRAIN_TEST_SPLIT, batch_size=batch_size)

  # Fixed values ( i.e.: not configurable)
  ALPHABET_SIZE = len(dataset.alphabet)
  INPUT_DIMENSION = max_image_height * frame_size

  print(f'Max image height {max_image_height}')
  print(f'Max image width {max_image_width}')
  print(f'ِAlphabet Size {ALPHABET_SIZE}')

  #============== Define and train the model
  model = OCRModel(INPUT_DIMENSION, hidden_layer_size, hidden_layer_num, ALPHABET_SIZE, dropout_ratio=dropout_ratio)

  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  start_time = time.time()

  losses = model.train_model(train_data, optimizer, epochs, clipping_value)
  print(f'--- Training took {(time.time() - start_time) / 60} minutes ---')
  print(f'Final Mean loss {np.mean(losses)} \n')

  # ============= Save the trained model
  torch.save(model.state_dict(), f'{model_name}_dict.pth')

  model.eval()

  return model.predict(train_data, dataset.alphabet), model.predict(test_data, dataset.alphabet), model.predict(holdout_data, dataset.alphabet)
