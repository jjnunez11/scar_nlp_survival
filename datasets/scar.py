from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.vocab import build_vocab_from_iterator
import io
import os.path
# from collections import Counter
# from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import random
import warnings


class SCAR:
    DATASET_NAME = "SCAR"
    NUM_CLASSES = 1

    def __init__(self, batch_size, data_root, target, undersample=False, device=torch.device('cuda:0'), min_freq=10):
        self.batch_size = batch_size
        self.device = device

        if undersample:
            self.data_dir = os.path.join(data_root, target + "_undersampled")
            self.n_lines['train'] = 1815
        else:
            self.data_dir = os.path.join(data_root, target)

        self.f_train = os.path.join(self.data_dir, 'train.tsv')
        self.f_dev = os.path.join(self.data_dir, 'dev.tsv')
        self.f_test = os.path.join(self.data_dir, 'test.tsv')
        self.f_dict = {'train': self.f_train,
                       'dev': self.f_dev,
                       'test': self.f_test}

        with open(self.f_train) as f:
            self.n_train = len(f.readlines())
        f.close()
        with open(self.f_dev) as f:
            self.n_dev = len(f.readlines())
        f.close()
        with open(self.f_test) as f:
            self.n_test = len(f.readlines())
        f.close()
        # from old setting
        warnings.warn("Warning, manually setting n_train")
        self.n_train = 30953


        self.n_lines = {'train': self.n_train,
                        'dev': self.n_dev,
                        'test': self.n_test}

        if undersample:
            self.n_lines['train'] = 1815

        # Make iters of the input text files
        self.train_iter, self.dev_iter, self.test_iter = self.create_iter(split=('train', 'dev', 'test'))

        # Convert training set to iter of tokens
        self.train_token_iter = self.get_scar_tokens(self.create_iter(split='train'))

        # Build vocabulary from tokens
        self.vocab = self.build_scar_vocab(min_freq)

    def get_vocab_size(self):
        return len(self.vocab)

    def build_scar_vocab(self, min_freq=10):
        vocab = build_vocab_from_iterator(self.train_token_iter, min_freq=min_freq,
                                          specials=('<BOS>', '<EOS>', '<PAD>'))

        # Add unknown token at default index position
        unknown_token = '<unk>'
        vocab.insert_token(unknown_token, 0)
        vocab.set_default_index(vocab[unknown_token])

        return vocab

    @staticmethod
    def tokenizer(t):
        our_tokenizer = get_tokenizer('basic_english')
        return our_tokenizer(t)

    @staticmethod
    def target_parse(target_text):
        """
        DELETE THIS, USE TARGET TRANSFORMER BELOW INSTEAD

        Compability with Hedwig requires that a binary target be two digits, to support multi-class.
        For now, keep it like this but convert it to single digit (0 or 1) at this point

        :param target_text:
        :return:
        """

        if len(target_text) == 2:
            print("Target is two digits, converting)")
            if target_text == "10":
                return 0
            elif target_text == "01":
                return 1
        elif len(target_text) == 1:
            print("Target is already only one digit!")
            return int(target_text)

    @_wrap_split_argument(('train', 'dev', 'test'))
    def create_iter(root, split):
        def generate_scar_data(key, files):
            f_name = files[key]
            f = io.open(f_name, "r")
            for line in f:
                values = line.split("\t")
                assert len(values) == 2, \
                    'Error: excepted SCAR datafile to be tsv format, but splitting by tab did not yield 2 parts'
                label = values[0]  # root.target_parse(values[0])
                text = values[1]
                yield label, text

        iterator = generate_scar_data(split, root.f_dict)
        return _RawTextIterableDataset(root.DATASET_NAME, root.n_lines[split], iterator)

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(self.label_transform(_label))
            processed_text = torch.tensor(self.text_transform(_text), device=self.device)
            text_list.append(processed_text)
        return pad_sequence(text_list, padding_value=3.0), torch.tensor(label_list, device=self.device)

    def batch_sampler(self, split):
        split_list = list(self.create_iter(split=split))
        indices = [(i, len(self.tokenizer(s[1]))) for i, s in enumerate(split_list)]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), self.batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + self.batch_size * 100], key=lambda x: x[1]))

        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), self.batch_size):
            yield pooled_indices[i:i + self.batch_size]

    def get_bucket_dataloader(self, split):
        split_list = list(self.create_iter(split=split))
        return DataLoader(split_list, batch_sampler=self.batch_sampler(split=split),
                          collate_fn=self.collate_batch)

    def train_dataloader(self):
        return self.get_bucket_dataloader('train')

    def dev_dataloader(self):
        return self.get_bucket_dataloader('dev')

    def test_dataloader(self):
        return self.get_bucket_dataloader('test')

    @classmethod
    def get_scar_tokens(cls, train_iter):
        """
        :param train_iter: SCAR training data iterator returning label and text for each example
        :return: Generator with each text line tokenized
        """
        for label, text in train_iter:
            yield cls.tokenizer(text)

    @staticmethod
    def label_transform(label):
        """
        Transforms labels, which may be in the hedwig format of 10 = 0, 01 = 1, to a binary float

        Will need to add more support to make multi-label, if we wanna go there.

        :param label: the string representation of the label, maybe '0'/'1' or '10'/'01'
        :return: float representation, 0 or 1. If need multi-label, will need to change to return a list etc.
        """

        # print(f'Here is a label: {label} with type {type(label)}')

        if len(label) == 1:
            return float(label)
        elif len(label) == 2 and label in ['10', '01']:
            if label == '10':
                return 0.0
            elif label == '01':
                return 1.0
        else:
            raise ValueError("Invalid target provided, current supports '0'/'1' or '10'/'01'")

    def text_transform(self, text):
        """
        Text transformer. Currently we do the hedwig preprocessing before any of this which is:
        string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)

        So then, all we need to do is convert a patient's text to tokens
        :param text: text of a patient
        :return: text after transformation to vocab index position
        """
        return [self.vocab['<BOS>']] + [self.vocab[token] for token in self.tokenizer(text)] + [self.vocab['<EOS>']]

    def get_class_balance(self):
        # print(f"Expecting 17692/30953 {17692/30953} for scar_emots")

        targets_equal_one = 0
        targets_total = 0

        for _, targets in self.train_dataloader():
            targets = targets.view(-1, 1).float()
            targets_total += len(targets.cpu().detach().numpy())
            targets_equal_one += targets.cpu().detach().numpy().sum()

        return targets_equal_one/targets_total

if __name__ == '__main__':
    scar = SCAR(2)

    # Check that iter and lists created
    print(f'train iter length {len(scar.train_iter)}')

    # Check vocab creation
    print("The length of the new vocab is", len(scar.vocab))

    check_token = scar.vocab.get_itos()
    print("The token at index 2 is", check_token[2])

    # Check that text transformer is working
    print("input to the text_transform:", "here is an example")
    print("output of the text_transform:", scar.text_transform("here is an example"))

    # Check that collate_batch is working
    train_dataloader = DataLoader(scar.create_iter(split='train'), batch_size=8,
                                  collate_fn=scar.collate_batch)
    # print(next(iter(train_dataloader)))

    # Finally, check that bucket_dataloader is working
    d = scar.dev_dataloader()
    print(len(scar.create_iter(split='dev')))
    print(next(iter(d)))
