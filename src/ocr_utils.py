import numpy as np
import glob
import torchvision
import os
import operator
import torch
import matplotlib.pyplot as plt

from Levenshtein import distance
from collections import Counter
from PIL import Image
from scipy.stats import entropy
from torchvision import transforms

def text2vec(text, alphabet, max_text_length):

    """
    Convert text to vector, the vector is of length max_text_length
    each element in the vector is the index of the corresponding char in the alphabet.

    e.g.: 'This is example' ~> [12, 34, 5, 8, 19, 5, 8, 2, 33, 1, 23, 22, 20, 2, 0, 0, 0]

    Args:
        text (str): Text to convert to vector
        alphabet (:obj: `set`): The alphabet used in the data set
        max_text_length (int): The length of the longest text in the data test
    Returns:
        (:obj: `list`): The vector representation of a text
    """

    # Use zeros because the blank symbol index is 0
    # so the we don't need to pad texts with length < max_text_length
    text_vec = np.zeros(max_text_length, dtype = int)
    
    for i in range(0,len(text)):
        text_vec[i] = np.where(alphabet == text[i])[0]
    
    return text_vec


def prob2text(model_output, alphabet):

    """
    Convert model output ( probabilities ) into  texts

    Args:
        model_output (array_like): The output of the model. Should be in form of (batch, sequence, probs)
        alphabet (:obj: `list`): The alphabet used in the data set

    Returns:
        (:obj: `list`): Represents the texts corresponding to the model output
    """
    batch_char_indices = np.argmax(model_output, axis = 2)
    text_output = []
    for sample_char_indices in batch_char_indices:
        text = ''.join(alphabet[sample_char_indices])
        text_output.append(text)
    
    return text_output

def show_image(image_tensor):
    """
    Display the image related to the tensor
    
    Args:
        image_tensor (array_like): A tensor that represent an image
    """
    to_pil = torchvision.transforms.ToPILImage()
    to_pil(image_tensor).show()


def clean_text(predicted_text):

    """
    Remove repeated chars and blank character from a tex

    Args:
        predicted_text (str): The text predicted by the mode

    Returns:
        (str): The final predcited text after clean up
    """
    
    final_text = predicted_text[0]
    
    for i in range(1, len(predicted_text)):
        
        if final_text[-1:] != predicted_text[i]:
            final_text+= predicted_text[i]

    return final_text.replace('$', '')


def get_text_statistics(base_dir):

    """
    Get some statistics about the text in the dataset

    Args:
        base_dir (str): The base directory of the dataset

    Returns:
        (:obj: `dict`): Dictionary Containing statistics about the text in the dataset
    """
    
    statistics = {}

    for subcorpora in os.listdir(base_dir):
        statistics[subcorpora] = {}
        alphabet = set()
        char_counter = Counter('')
        
        books_path = os.path.join(base_dir, subcorpora)
                
        for book in os.listdir(books_path):
            full_path = os.path.join(books_path, book)
            text_files = glob.glob(full_path + "/*/*.txt")
                
            for text_file in text_files:
                with open(text_file,encoding = 'utf-8') as f:
                    text = f.read().rstrip()
                    char_counter = char_counter + Counter(text)
                    char_set = set(' '.join(text))
                    alphabet = alphabet.union(char_set)

            statistics[subcorpora]['alphabet'] = alphabet
            statistics[subcorpora]['char_counter'] = char_counter

    return statistics

def create_hist(char_count, top_chars = 10):
    """
    Create a histogram out of an array

    Args:
        char_count (:obj: `list`): The frequency of the chars
        top_chars (int): How many bars we want to have in the histogram

    Returns:
        (:obj: `list`, :obj: `list`): The chars with top frequency and their related values
    """

    sorted_char_count = sorted(char_count.items(), key=operator.itemgetter(1))
    sorted_char_count.reverse()
    keys = [item[0] for item in sorted_char_count ]
    values = [item[1] for item in sorted_char_count ]
    
    return keys[:top_chars], values[:top_chars]

def plot_hist(text_statistics, subcorpora_name,  axis):
    """
    Plot a histogram of char count for specific subcorpora

    Args:
        text_statistics (:obj: `dict`): Statistics about all subcorporas in the dataset
        subcorpora_name (str): The subcorpora name to get its histogram
        axis (:obj: `axis`): An axis to plot the histogram on
    """
    keys, values = create_hist(text_statistics[subcorpora_name]['char_counter'], 20)
    axis.bar(keys, values, 0.5, color='b')
    axis.set_title(subcorpora_name)


def print_predicted_text(predictions, alphabet, samples_number = 10 ):
    """
    Print some samples from the model predictions

    Args:
        predictions (:obj: `dict`): The final model predictions and the ground truth
        samples_number (int): How many samples to print
    """
    for true_text, predicted_text, text_confidence, chars_confidence in zip(predictions['true_texts'], predictions['predicted_texts'], predictions['texts_confidence'], predictions['chars_confidence']):
        print(f'Original Text:   {true_text}')
        print(f'Predicted Text:  {predicted_text} Confidence: {text_confidence}')
        
        s=""
        for char_confidence in chars_confidence:
            s+=f'{alphabet[char_confidence[0]]}: {format(char_confidence[1], ".2f")} '
        print(f'{s} \n')

        samples_number-=1

        if samples_number ==0: break

def get_prediction_error(predictions):
    """
    Calculate the final Levenshtein error in the model predictions

    Args:
        predictions (:obj: `dict`): The final model predictions and the ground truth

    Returns:
        (float): The mean Levenshtein error in the model predictions
    """
    errors = []
    
    for true_text, predicted_text in zip(predictions['true_texts'], predictions['predicted_texts']):
        error = distance(true_text, predicted_text) / len(true_text)
        errors.append(error)

    return np.mean(errors)


def explain(x, model):

    """
    Display an imahge from the input space that represents a gradient based explanation of the model output

    Args:
        x (array_like): The text line image to explain its model result
        model (:obj: `OCRModel`): The trained OCR model
    """
    
    x.requires_grad_()
    output = model(x)
    
    frames_predictions = torch.zeros(output.shape)
    max_indx = torch.argmax(output, dim=2)
    
    frames_predictions[0,:, max_indx] = output[0,:, max_indx]

    output.backward(frames_predictions)
    explanation = (x.grad[0].t()**2).data

    plt.figure(figsize=(15, 8))
    plt.imshow(explanation, cmap='seismic', vmin=-abs(explanation).max(), vmax=abs(explanation).max())

def get_confusion_matrix(true_texts, predicted_texts, chars, padding_char):

    """
    Build the confusion matrix from the predicted texts

    Args:
        true_texts (array_like): The target texts
        predicted_texts (array_like): The texts predicted by the model
        chars (:obj: `list`): The alphabet of the predicted and true texts
        padding_char (char): A char to use in padding the texts with different lengths

    Returns:
        (array_like): The confusion matrix
    """
   
    confusion = np.zeros((len(chars), len(chars)))
    iso = 0
    
    for true_text, predicted_text in zip(true_texts, predicted_texts):
        #pad it to overcome the case of shorter predicted text than true text
        #predicted_text = predicted_text.ljust(len(true_text), padding_char)
        
        if len(true_text) != len(predicted_text): continue

        for i, char in enumerate(true_text):
            row = chars[char]
            col = chars[predicted_text[i]]
            confusion[row,col]+= 1

    return confusion


def calculate_sample_confidence_char_wise(indices, probs):

    """
    Calculate the confidence of one sample based on its chars

    Args:
        indices (:obj: `list`): The indices of the predictions
        probs (:obj: `list`): The probability associated with each predcition

    Returns:
        (float, :obj: `dict`): The sample confidence and each of its chars confidence
    """

    chars_confidence = []
    chars_confidence_map = []
    entropies = entropy(probs.T, base= probs.shape[1])

    i = 0
    while i < len(indices):
        if indices[i]!=0:
            char_confidences = []
            char_confidences.append(1 - entropies[i])
            char_index = indices[i]

            while i+1 < len(indices) and indices[i]==indices[i+1]:
                i+=1
                char_confidences.append(1 - entropies[i])

            char_confidence= np.array(char_confidences).mean()
            chars_confidence.append(char_confidence)
            chars_confidence_map.append((char_index,char_confidence))
        i+=1


    return np.array(chars_confidence).mean(), chars_confidence_map

def calculate_sample_confidence_frame_wise(indices, probs):

    """
    Calculate the confidence of one sample based on its frames

    Args:
        indices (:obj: `list`): The indices of the predictions
        probs (:obj: `list`): The probability associated with each predcition

    Returns:
        (float): The sample confidence
    """

    frames_confidence = []
    entropies = entropy(probs.T, base= probs.shape[1])
    
    #Remove blank character index
    entropies = entropies[indices!=0] 

    return (1 - entropies).mean()

def calculate_confidence(predictions):

    """
    Calculate the confidence of the predicited texts and the confidence of each char in each text sample

    Args:
        predictions (:obj: `list`): The predicted texts

    Returns:
        (:obj: `list`, :obj: `list`): The texts confidence and the chars confidences
    """

    probs = np.exp(predictions)
    max_probs_indices = np.argmax(predictions, axis = 2)
    texts_confidence = []
    combined_chars_confidence = []

    for i in range(0,max_probs_indices.shape[0]):
        confidence, chars_confidence = calculate_sample_confidence_char_wise(max_probs_indices[i], probs[i])
        texts_confidence.append(confidence)
        combined_chars_confidence.append(chars_confidence)

    return texts_confidence, combined_chars_confidence
