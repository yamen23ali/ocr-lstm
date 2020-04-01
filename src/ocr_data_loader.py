import os

from ocr_utils import *
from ocr_data_set import GT4HistOCRDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class OCRCVDataLoader(object):

    """ 
        This class is used to do load the dataset in a way that enables cross validation

        Args:
            base_dir (str): The base directory of that contains the dataset
            dataset_name (str): The name of the dataset to load
            transformation (:obj:`list` of Transform): List of transofrmations to apply on the loaded images
            blank_symbol (char): The char to use as a blank symbol
            batch_size (int): The number of samples in each batch
            k (int): The number of folds in the k-fold algorithm

        Attributes:
            batch_size (int): The number of samples in each batch
            k (int): The number of folds in the k-fold algorithm
            dataset (:obj: `GT4HistOCRDataset`): The number of samples in one batch
            batch_size (int): The number of samples in one batch
            splits (:obj: `list` of int): The indices of the splits for the k-fold algorithm
    """

    def __init__(self, base_dir = '/home/space/datasets/text/GT4HistOCR', dataset_name= None,
        transformation=transforms.Compose([transforms.ToTensor()]), blank_symbol='$', batch_size=10, k = 5):

        self.batch_size = batch_size
        self.k = k

        dataset_dir = os.path.join(base_dir, dataset_name)
        self.dataset = GT4HistOCRDataset(dataset_dir, transformation, blank_symbol)
        
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        self.splits = self.__split_indices__(indices, k)

    def get_dataset(self):
        """
            Get the loaded dataset

            Returns:
                (:obj: `GT4HistOCRDataset`): An instance of the loaded dataset
        """
          
        return self.dataset

    def load_data(self, test_split_ind):
        """
            Create training and test data loader based on the split index to retrieve data through them

            Args:
                test_split_ind (int): The index of the test split

            Returns:
                (:obj: `DataLoader`, :obj: `DataLoader`): The train and test data loaders
        """
        
        test_indices = self.splits[test_split_ind]
        train_indices = [x for i,x in enumerate(self.splits) if i!=test_split_ind]
        train_indices = np.array(train_indices).flatten()

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataloader = DataLoader(self.dataset, batch_size= self.batch_size,
                        shuffle=False, num_workers=4,  sampler=train_sampler)

        test_dataloader = DataLoader(self.dataset, batch_size= self.batch_size,
                        shuffle=False, num_workers=4,  sampler=test_sampler)

        return train_dataloader, test_dataloader

    def __split_indices__(self, indices, splits):
        """
            Split the data indices into multiple parts ( i.e. splits) to use in k-fold algorithm

            Args:
                splits (int): The Number of splits
                indices (:obj: `list` of int): List of 

            Returns:
                (:obj: `list` of int): A list of splits indiceis
        """
        split_length = int(len(indices) / splits)

        return [ indices[i*split_length : (i+1)*split_length] for i in range(splits) ]


def load_data(base_dir = '/home/space/datasets/text/GT4HistOCR', dataset_name= None,
    holdout_book = None, transformation=transforms.Compose([transforms.ToTensor()]), 
    train_test_split=0.8, blank_symbol='$', batch_size=10):

    """
    Create training and test loaders that depends on GT4HistOCRDataset
     
    Args:
        base_dir (str): The base directory for the data set
        dataset_name (str): The data set folder name inside the base directory
        holdout_book (str): The name of the book to use as holdout, later we perform evaluation on this book
        transformation (transforms): a composed transformation to apply on the images
        train_test_split (float): The ratio to split the data with. The default is splitting the data
                with (80%) of it as training and the remaining (20%) as test
        blank_symbol (str): The symbol to represent blank chars (in CTC )

    Returns:

        (:obj: `DataLoader`, :obj: `DataLoader`, :obj: `DataLoader`, :obj: `GT4HistOCRDataset`): The data loaders to use during training, testing and final evaluation and the dataset of GT4Hist
    """
    
    dataset_dir = os.path.join(base_dir, dataset_name)
    
    dataset = GT4HistOCRDataset(dataset_dir, transformation, blank_symbol)

    # Get indices of the samples coming from the holdout book
    holdout_indices = dataset.get_book_indices(holdout_book)
    print(len(holdout_indices))
    
    # Get indices of the remaining samples
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_test_indices = [x for x in indices if x not in holdout_indices]
    print(len(train_test_indices))

    # Split the remaining smaples to train and test
    split_point = int(np.floor(train_test_split * len(train_test_indices)))
    np.random.shuffle(train_test_indices)
    train_indices, test_indices = train_test_indices[:split_point], train_test_indices[split_point:]
    
    # Create samplers for (train, test, holdout) sets
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    holdout_sampler = SubsetRandomSampler(holdout_indices)

    
    # Create loaders for (train, test, holdout) sets
    train_dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4,  sampler=train_sampler)
    
    test_dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4,  sampler=test_sampler)

    holdout_dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4,  sampler=holdout_sampler)
    
    return train_dataloader, test_dataloader, holdout_dataloader, dataset
