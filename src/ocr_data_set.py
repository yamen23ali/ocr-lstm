import torch
import glob
import numpy as np
import re

from ocr_utils import text2vec
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

class GT4HistOCRDataset(Dataset):

    """ 
        This class is used to build GT4HistOCRDataset from the raw images and files.

        Args:
            dataset_dir (str): Path to the data set
            transformation (:obj:`Transform`): A composed transformation to apply on the images
            blank_symbol (str): The symbol to represent blank chars (in CTC )

        Attributes:
            dataset_dir (str): Path to the data set
            transformation (:obj:`Transform`): A composed transformation to apply on the images
            blank_symbol (str): The symbol to represent blank chars (in CTC )

            image_files_names (:obj: `list` of str): The names of the images files
            text_files_names (:obj: `list` of str): The names of the text files
            alphabet (:obj: `Set`): The alphabet contained in the dataset
            max_text_length (int): The maximum text length in the dataset
    """
    def __init__(self, dataset_dir, transformation, blank_symbol):
        self.dataset_dir = dataset_dir
        self.image_files_names = glob.glob(dataset_dir + "/*/*.png")
        self.text_files_names = glob.glob(dataset_dir + "/*/*.txt")
        self.transformation = transformation
        self.blank_symbol = blank_symbol

        self.alphabet = self.__get_alphabet()
        self.max_text_length = self.__get_max_text_length()

    def get_book_indices(self, book_name):
        """
        Get the file indices that belong to specific book

        Args:
            book_name (str): The book to get its file indices

        Returns:
            (:obj: `list` of int): The files indices of the specified book
        """
        return [ i for i, file_path in enumerate(self.text_files_names) if book_name in file_path]
    
    def __get_text(self, text_file):
        """
        Get text from file and clean it up

        Returns:
            (str): Text ready for usage
        """
        with open(text_file,encoding = 'utf-8') as f:
                text = f.read().rstrip()
        return text

    def __get_alphabet(self):
        """
        Extract alphabet from data set text files and add the blank symbol to it

        Returns:
            (:obj: `Set`): The alphabet used in the data set
        """
        alphabet = set()

        for text_file in self.text_files_names:
            text = self.__get_text(text_file)
            char_set = set(' '.join(text))
            alphabet = alphabet.union(char_set)

        # The blank symbol must be at index 0
        alphabet = list(alphabet)
        alphabet.sort()
        alphabet.insert(0, self.blank_symbol)

        return np.array(alphabet)

    def __get_max_text_length(self):
        """
        Get the max length of texts in the data set

        Returns:
            (int): The length of the longest text in the data test
        """
        max_len = 0

        for text_file in self.text_files_names:
            text_len = len( self.__get_text(text_file) )
            
            if text_len > max_len: max_len = text_len
        return max_len

    def __len__(self):
        """
        Get the dataset size
        Returns:
            (int): data set length ( i.e.: number of samples)
        """
        return len(self.image_files_names)

    def __getitem__(self, idx):
        """
        Return a specific item from the dataset

        Args:
            idx (int): The index number of the training sample to get

        Returns:
            (:obj: `dict`): A dictionary that contains the sample (image, text, text_length, text_vector)
        """
        
        # read image, convert to B&W and apply a compose transformation on it
        image_file = self.image_files_names[idx] 
        raw_image = Image.open(image_file) #.convert('1')
        
        image = self.transformation(raw_image)
        
        # text file should have same name as its corresponding image file, but with different suffix
        match = re.match("(.*\/\d+)\..*(\.png)", image_file)
        text_file = match.group(1) + '.gt.txt'
        
        text = self.__get_text(text_file)

        text_vector = text2vec(text, self.alphabet, self.max_text_length)
        
        
        return {'image': image,  'text': text,  'text_length': len(text), 'text_vector': text_vector}
