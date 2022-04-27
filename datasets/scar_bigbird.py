from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _RawTextIterableDataset
import io
import os.path
# from collections import Counter
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import random
from transformers import BigBirdTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import re
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class SCARBigBirdDataset(Dataset):
    def __init__(self, consults, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.consults = consults
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.consults)

    def __getitem__(self, item_idx):
        consult = self.consults[item_idx]
        inputs = self.tokenizer(
            consult,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        # print(f'len of labels is {len(self.labels)} while the item_idx is {item_idx}')
        #print(f'Len of input_ids {len(input_ids)} and attn mask {len(attn_mask)} and len of consult: {len(consult)}')
        #print(f'Here is the first 10 words {consult[100:110]}')
        #sequence = ' '.join(consult)
        #tokenized_seq = self.tokenizer.tokenize(sequence)
        #print(f'Here is the first 10 words tokenized {self.tokenizer.tokenize(sequence)}')
        #inputs_idx = inputs['input_ids']
        #print(f'Here is the length of raw consult {len(consult)} and after tokenizing {len(tokenized_seq)} and of inputs input_ids {inputs_idx.size()}')

        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'label': torch.tensor(self.labels[item_idx], dtype=torch.float)
        }


class SCAR_BigBird(pl.LightningDataModule):
    NUM_LINES = {'train': 30953,
                 'dev': 4459,
                 'test': 4458}
    DATASET_NAME = "SCAR"
    NUM_CLASSES = 1

    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.max_len = config.max_tokens
        self.device = config.device
        self.data_dir = os.path.join(config.data_dir, config.target)
        self.f_train = os.path.join(self.data_dir, 'train.tsv')
        self.f_dev = os.path.join(self.data_dir, 'dev.tsv')
        self.f_test = os.path.join(self.data_dir, 'test.tsv')
        self.debug = config.debug

        self.tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

        # Clean and prepare the raw data
        self.raw_x_train, self.raw_y_train = self.clean_data(self.f_train)
        self.raw_x_dev, self.raw_y_dev = self.clean_data(self.f_dev)
        self.raw_x_test, self.raw_y_test = self.clean_data(self.f_test)

        # Tokenize up to max length, and put into a PyTorch Dataset
        self.train_dataset = SCARBigBirdDataset(consults=self.raw_x_train,
                                             labels=self.raw_y_train,
                                             tokenizer=self.tokenizer,
                                             max_len=self.max_len)
        self.dev_dataset = SCARBigBirdDataset(consults=self.raw_x_dev,
                                           labels=self.raw_y_dev,
                                           tokenizer=self.tokenizer,
                                           max_len=self.max_len)
        self.test_dataset = SCARBigBirdDataset(consults=self.raw_x_test,
                                            labels=self.raw_y_test,
                                            tokenizer=self.tokenizer,
                                            max_len=self.max_len)

    def clean_data(self, f):
        """
        Reads in the raw data file, and outputs two lists of the corresponding cleaned text and arrayed y

        :param f: train, dev, or test raw data files
        :return: cleaned and prepared list of cleaned text, and of the y in 0 -> [1 0], 1 -> [0 1] format
        """

        texts = []
        labels = []

        # Open file
        file = open(f, "r")

        i = 0
        for line in tqdm(file, desc=f'Reading in {f}'):
            values = line.split("\t")
            assert len(values) == 2, "Reading a file, we found a line with multiple tabs"
            raw_label, raw_text = values[0], values[1]

            texts.append(self.text_transform(raw_text))
            labels.append(self.label_transform(raw_label))

            i += 1

            if self.debug:
                if i >= 99:
                    break

        file.close()

        return texts, labels

    @staticmethod
    def text_transform(text):

        # fetch alphabetic characters
        text = re.sub("[^a-zA-Z]", " ", text)
        # TODO: Probably can eventually remove this when fix the pre-processing

        # convert text to lower case
        text = text.lower()

        # split text into tokens to remove whitespaces
        tokens = text.split()

        return " ".join(tokens)

    @staticmethod
    def label_transform(label_text):
        # For now, use a single array, but could make this binary

        if len(label_text) == 1 and float(label_text) == 0:
            # return [1, 0]
            return [0]
        elif len(label_text) == 1 and float(label_text) == 1:
            # return [0, 1]
            return [1]
        elif len(label_text) == 2 and label_text in ['10', '01']:
            if label_text == '10':
                # return [1, 0]
                return [0]
            elif label_text == '01':
                # return [0, 1]
                return [1]
        else:
            raise ValueError("Invalid target provided, current supports '0'/'1' or '10'/'01'")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):  # This is called val, on dev, in PyTorch lightning
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

    def get_class_balance(self):
        #  print(f"Expecting 17692/30953 {17692/30953} for scar_emots")
        targets_equal_one = sum(x[0] for x in self.raw_y_train)
        targets_total = len(self.raw_y_train)
        #  print(f"Found {targets_equal_one}/{targets_total} or {round(targets_equal_one/targets_total, 3)}")

        return targets_equal_one/targets_total

    def get_n_training(self):
        return len(self.raw_y_train)
