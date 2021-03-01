# -*- coding: utf-8 -*-
import pandas as pd

from test_tube import HyperOptArgumentParser
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import re, pickle, torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

class RedditDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_csv = 'file.csv', aux_task = 'group', le = None, le_aux = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.comments = pd.read_csv(data_csv)
        if aux_task == 'None':
            aux_task = 'group'
        if aux_task == 'emotions':
            #self.comments.loc[self.comments[aux_task].isna(), aux_task] = 'Neutral'
            self.comments['label_aux'] = self.comments[['Anger','Contempt','Disgust','Fear','Hope','Pride','Sympathy','Emotions_Neutral']].values.tolist()
            self.columns = list(self.comments[['Anger','Contempt','Disgust','Fear','Hope','Pride','Sympathy','Emotions_Neutral']].columns)
        else:
            le_aux.fit(self.comments[aux_task].values)
            self.comments['label_aux'] = le_aux.transform(self.comments[aux_task].values)

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return self.comments.iloc[idx][['body', 'label_aux', 'group', 'bias', 'usVSthem_scale']]


def pad_seq(seq, max_batch_len, pad_value):
    # IRL, use pad_sequence
    # https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pad_sequence.html
    return seq + (max_batch_len - len(seq)) * [pad_value]

class MyCollator(object):
    def __init__(self, model_name, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    def __call__(self, batch):
        output = {}
        texts = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'LINK', comment['body'], 
                        flags=re.MULTILINE) for comment in batch]
        tokenized = self.tokenizer(texts, padding='longest', truncation=True, max_length = self.max_length, return_tensors = 'pt', add_special_tokens = True)
        output['labels'] = torch.tensor([element['usVSthem_scale'] for element in batch], dtype=torch.float)
        output['labels_aux'] = torch.tensor([element['label_aux'] for element in batch], dtype=torch.float)
        return tokenized.data, output


def sentiment_analysis_dataset(
    hparams: HyperOptArgumentParser, train=True, val=True, test=True
):
    """
    Loads the Dataset from the csv files passed to the parser.
    :param hparams: HyperOptArgumentParser obj containg the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.

    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """
    if train:
        dataset = RedditDataset(hparams.train_csv, hparams.aux_task, hparams.le, hparams.le_aux)
    if val:
        dataset = RedditDataset(hparams.dev_csv, hparams.aux_task, hparams.le, hparams.le_aux)
    if test:
        dataset = RedditDataset(hparams.test_csv, hparams.aux_task, hparams.le, hparams.le_aux)
    return dataset
