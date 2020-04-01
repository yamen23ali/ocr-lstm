import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import time

from ocr_utils import *
from torch import nn

class OCRModel(nn.Module):
    """ 
        This class represents the BLSTM model that is used for the OCR problem

        Args:
            input_dim (int): Number of neurons in the input layer
            hidden_dim (int): Number of neurons in the hidden layer
            layer_dim (int): Number of hidden layers
            output_dim (int): Number of neurons in the output layer
            dropout_ratio (float): The probability of dropping a neuron out during the training

        Attributes:
            hidden_dim (int): Number of neurons in the hidden layer
            layer_dim (int): Number of hidden layers
            lstm (:obj: `nn.LSTM`): LSTM neural network
            fc (:obj: `nn.Linear`): Linear neural network layer
    """
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_ratio=0.0):
        super(OCRModel, self).__init__()
        
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, layer_dim, batch_first=True, bidirectional=True)


        self.drop_out = nn.Dropout(p=dropout_ratio)

        # Readout layer
        self.fc = nn.Linear(self.hidden_dim*2, output_dim)

        self.__prepare_for_device__()

    def forward(self, x):
        """
        Make a forward pass in the neural network

        Args:
            x (array_like): Batch of text line images frames

        Returns:
            (array_like): Probability vectors of the fed frames in the input
        """

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).to(self.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.drop_out(out)

        out = self.fc(out)

        return F.log_softmax(out, dim=2)
    
    def train_model(self, training_data, optimizer, epochs=10, clipping_value=10):
        """
        Train this model using teh passed training data and optimizer

        Args:
            training_data (:obj: `DataLoader`): The data to use in training
            optimizer (:obj: `optim.SGD`): The optimizer to use in the training
            epohcs (int): The number of iterations in the SGD algorithm
            clipping_value (float): The value to clip the grdient at

        Returns:
            (array_like): Losses during the training
        """

        for epoch in range(0,epochs):
            losses = list()

            for batch in enumerate(training_data):
                
                # Move data to GPU
                images_batch = batch[1]['image'].to(self.device)
                text_vector_batch = batch[1]['text_vector'].to(self.device)
                text_length_batch = batch[1]['text_length'].to(self.device)

                # Clear gradients
                optimizer.zero_grad()

                probabilities = self(images_batch).permute(1,0,2)
                
                # Compute crossentropy loss
                ctc_loss = nn.CTCLoss(zero_infinity=True)
                probabilities_lengths = torch.full((probabilities.shape[1],), 
                                                   probabilities.shape[0], dtype=torch.long).to(self.device)
                
                output = text_vector_batch.clone().detach().requires_grad_(False).to(self.device)
                output_lengths = text_length_batch.clone().detach().requires_grad_(False).to(self.device)
                
                loss = ctc_loss(probabilities, output, probabilities_lengths, output_lengths).to(self.device)
                
                # Compute gradient
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), clipping_value)
                
                # Perform gradient descent
                optimizer.step()

                # Track losses
                losses.append(loss.item())
            
            print("Epoch {}, Mean loss {}".format(epoch, np.mean(losses)))
        
        return losses

    def predict_batch(self, images_batch, alphabet):
        """
        Get the model prediction for a batch of text line images

        Args:
            images_batch (array_like): The text line images to get the model prediction for
            alphabet (:obj: `set`): The alphabet of the dataset

        Returns:
            (:obj: `list`, :obj: `list`, :obj: `list`): The text predictions for the provided images and their related confidences on sample and char level
        """
        output = self(images_batch.to(self.device)).cpu().detach().numpy()
        texts_confidence, chars_confidence = calculate_confidence(output)
        
        return prob2text(output, alphabet), texts_confidence, chars_confidence
    
    def predict(self, data, alphabet):
        """
        Get the model predictions along with the ground truth

        Args:
            data (:obj: `DataLoader`): The data we want to get the predictions for
            alphabet (:obj: `set`): The alphabet of the dataset

        Returns:
            (:obj: `dict`): The model predictions alongside the ground truth texts
        """
        predictions = {'true_texts':[], 'predicted_texts':[], 'texts_confidence':[], 'chars_confidence':[]}

        for batch in enumerate(data):
            true_texts = batch[1]['text']
            predicted_texts, texts_confidence, chars_confidence = self.predict_batch(batch[1]['image'], alphabet)

            for true_text, predicted_text, text_confidence, chars_confidence in zip(true_texts, predicted_texts, texts_confidence, chars_confidence):
                final_text = clean_text(predicted_text)
                predictions['true_texts'].append(true_text)
                predictions['predicted_texts'].append(final_text)
                predictions['texts_confidence'].append(text_confidence)
                predictions['chars_confidence'].append(chars_confidence)

        return predictions

    def __prepare_for_device__(self):
        """
        Prepare the model to run on cuda in case it is available
        """
        
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            device_num = random.randint(0, count-1)
            self.device = 'cuda:' + str(device_num)
            self.to(self.device)
        else:
            self.device = 'cpu'
