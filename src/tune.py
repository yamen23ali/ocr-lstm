import sys

import os
import torch
import glob
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import argparse
import base64

from ocr_data_loader import load_data
from ocr_utils import *
from ocr_image_transformations import *
from ocr_model import OCRModel
from torchvision import transforms
from torch import nn
import torchvision
from PIL import Image
import hashlib

#============= Get and Parse Arguments

parser=argparse.ArgumentParser()

parser.add_argument('--BASE_DIR', help='Base Directory Of The Dataset')
parser.add_argument('--DATA_SET_NAME', help='Data Set Name')
parser.add_argument('--HOLDOUT_BOOK', help='The Book we want to use as holdout')
parser.add_argument('--PARAM_NAME', help='Hyper Parameter name [lr,m,b]')


args=parser.parse_args()

# ========= Create model name
print(f'Model parameters {args}')

BASE_DIR = args.BASE_DIR
DATA_SET_NAME = args.DATA_SET_NAME
HOLDOUT_BOOK = args.HOLDOUT_BOOK

args_bytes = str(args).encode('utf-8')
args_hash= hashlib.sha1(args_bytes).hexdigest()
MODEL_NAME= f'model_{args_hash}'

print(f'Model Name : {MODEL_NAME}')

#========= Hyper parameters 

PARAM_NAME = str(args.PARAM_NAME)
# Training parameters
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
BATCH_SIZE = 50
DROP_OUT_RATIO= 0.0
FRAME_SIZE = 10
HIDDEN_LAYER_SIZE = 300
HIDDEN_LAYERS_NUM = 1 
EPOCHS = 100
TRAIN_TEST_SPLIT = .9
CLIPPING_VALUE = 100

#========= Hyper parameters 

def setup_model(lr, momentum, batch_size, hidden_layer_size, hidden_layer_num):
    MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT = 500, 20

    transformation = transforms.Compose([
        FixLineImage(),
        ImageThumbnail(MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH),
        transforms.ToTensor(),
        ImageTensorPadding(MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH),
        UnfoldImage(1, FRAME_SIZE, FRAME_SIZE)
    ])

    train_data, test_data, holdout_data, dataset = load_data(base_dir = BASE_DIR, dataset_name = DATA_SET_NAME,
                                           holdout_book=HOLDOUT_BOOK, transformation=transformation,
                                           train_test_split=TRAIN_TEST_SPLIT, batch_size=BATCH_SIZE)

    # Fixed values ( i.e.: not configurable)
    ALPHABET_SIZE = len(dataset.alphabet)
    INPUT_DIMENSION = MAX_IMAGE_HEIGHT * FRAME_SIZE

    print(f'Max image height {MAX_IMAGE_HEIGHT}')
    print(f'Max image width {MAX_IMAGE_WIDTH}')
    print(f'Alphabet Size {ALPHABET_SIZE}')

    # Define and train the model
    model = OCRModel(INPUT_DIMENSION, HIDDEN_LAYER_SIZE, HIDDEN_LAYERS_NUM, ALPHABET_SIZE, dropout_ratio=DROP_OUT_RATIO)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    return train_loader, test_loader, dataset, model, optimizer


def tune_model(hyper_param_list, param_name):
    all_loses = []
    print('Tuning Learning ', param_name)
    for hyper_param_val in hyper_param_list:
        if param_name == 'lr':
            train_loader, test_loader, dataset, model, optimizer = setup_model(hyper_param_val, MOMENTUM, BATCH_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYERS_NUM)
        elif param_name == 'm':
            train_loader, test_loader, dataset, model, optimizer = setup_model(LEARNING_RATE, hyper_param_val, BATCH_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYERS_NUM)
        elif param_name == 'b':
            train_loader, test_loader, dataset, model, optimizer = setup_model(LEARNING_RATE, MOMENTUM, hyper_param_val, HIDDEN_LAYER_SIZE, HIDDEN_LAYERS_NUM)

        losses = model.train(train_loader, optimizer, FRAME_SIZE, EPOCHS, CLIPPING_VALUE)
        print(losses)
        all_loses.append(losses)
    print('all loses', all_loses)
    min_lose_arg = np.argmin(np.array(all_loses))
    print('Best {} is {}', format(param_name, hyper_param_list[min_lose_arg]))

    
    
    
lrs = [1e0, 1e-2, 1e-4, 1e-6, 1e-8]
momenta = [0.0, 0.2, 0.5, 0.7, 0.9]
batch_sizes = [200, 300, 400, 450] 

if PARAM_NAME == 'lr':
    tune_model(lrs, PARAM_NAME)
elif PARAM_NAME == 'm':
    tune_model(momenta, PARAM_NAME)
elif PARAM_NAME == 'b':
    tune_model(batch_sizes, PARAM_NAME)
