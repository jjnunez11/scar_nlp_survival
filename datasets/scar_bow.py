import os
import pandas as pd
import numpy as np
from datasets.scar import SCAR
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import _pickle as pickle
import bz2


class SCARBoW:
    def __init__(self, config, undersample=False):
        self.max_tokens = config.max_tokens
        self.use_idf = config.use_idf
        self.data_dir = os.path.join(config.data_dir, config.target)

        if undersample:
            self.data_dir = os.path.join(config.data_dir, config.target + "_undersampled")
            # self.NUM_LINES['train'] = 1815
        else:
            self.data_dir = os.path.join(config.data_dir, config.target)

        self.train_file = os.path.join(self.data_dir, f'train_bow_{self.max_tokens}.csv')
        self.dev_file = os.path.join(self.data_dir, f'dev_bow_{self.max_tokens}.csv')
        self.test_file = os.path.join(self.data_dir, f'test_bow_{self.max_tokens}.csv')

        if not (os.path.exists(self.train_file) and
                os.path.exists(self.dev_file) and
                os.path.exists(self.test_file)):
            # If files don't exist, generate anew
            # Instantiate data frames to store the data
            self.raw_train_data = pd.DataFrame(columns=['label', 'text', 'vector'])
            self.raw_dev_data = pd.DataFrame(columns=['label', 'text', 'vector'])
            self.raw_test_data = pd.DataFrame(columns=['label', 'text', 'vector'])

            # Read the files to extract target, and tokenize the text
            self.raw_train_data = self.read_labels_and_tokens('train')
            self.raw_dev_data = self.read_labels_and_tokens('dev')
            self.raw_test_data = self.read_labels_and_tokens('test')

            # Using the tokens, make BoW vectors, using TF or TF-IDR, and then print out
            self.vectorize_tokens()

        # Read in data
        self.train_data = pd.read_csv(self.train_file)
        self.dev_data = pd.read_csv(self.dev_file)
        self.test_data = pd.read_csv(self.test_file)

    def vectorize_tokens(self):
        # Fit Vectorizer to training data, and then use to transform for dev and test
        vectorizer = CountVectorizer(max_features=self.max_tokens,
                                     tokenizer=StemTokenizer(),  # Tokenizes, stems, and remove stop words
                                     lowercase=True)

        train_counts = vectorizer.fit_transform(self.raw_train_data['text'])
        dev_counts = vectorizer.transform(self.raw_dev_data['text'])
        test_counts = vectorizer.transform(self.raw_test_data['text'])

        # Save vectorizer object for interpretation
        vectorizer_filename = os.path.join(self.data_dir, "vectorizer.bz2")
        with bz2.BZ2File(vectorizer_filename, 'w') as f:
            pickle.dump(vectorizer, f)

        # Fit TF-IDF-er (Term Frequency times inverse document frequency) on training data,
        # and then use to transform for dev and test
        tfidf_transformer = TfidfTransformer(use_idf=self.use_idf).fit(train_counts)
        self.raw_train_data['vector'] = tfidf_transformer.transform(train_counts).todense().tolist()
        self.raw_dev_data['vector'] = tfidf_transformer.transform(dev_counts).todense().tolist()
        self.raw_test_data['vector'] = tfidf_transformer.transform(test_counts).todense().tolist()

        # Save to csv for loading
        self.raw_train_data.to_csv(self.train_file)
        self.raw_dev_data.to_csv(self.dev_file)
        self.raw_test_data.to_csv(self.test_file)

    def read_labels_and_tokens(self, split):
        filename = os.path.join(self.data_dir, split + '.tsv')

        if split == 'train':
            df = self.raw_train_data
        elif split == 'dev':
            df = self.raw_dev_data
        elif split == 'test':
            df = self.raw_test_data
        else:
            df = pd.DataFrame()  # This is just to suppress a warning, will trigger error

        # Open file
        file = open(filename, "r")

        i = 0
        for line in tqdm(file):
            values = line.split("\t")
            assert len(values) == 2, "Reading a file, we found a line with multiple tabs"
            label, raw_text = values[0], values[1]
            df.loc[i, 'label'] = SCAR.label_transform(label)
            df.at[i, 'text'] = raw_text  # Use at so can accept a list

            i += 1

        file.close()

        return df

    @staticmethod
    def tokenize_text(text):
        """
        Processes and tokenizes the raw text before vectorization/tl-df/etc.
        Same applied to drain, dev, and test data

        :param text: One consult's input text
        :return: tokenized text
        """
        # Make lower case
        text = str.lower(text)
        # Tokenize
        tokenized_text = word_tokenize(text)
        # Remove stopwords
        tokenized_text_wo_sw = [word for word in tokenized_text if word not in stopwords.words('english')]
        # Stem
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        stemmed_text = [stemmer.stem(word) for word in tokenized_text_wo_sw]

        return ' '.join(stemmed_text)  # Return as a single string.

    def get_train_data(self):
        return self.train_data

    def get_dev_data(self):
        return self.dev_data

    def get_test_data(self):
        return self.test_data


class StemTokenizer:
    def __init__(self):
        self.sbs = SnowballStemmer("english", ignore_stopwords=True)

    def __call__(self, doc):
        return [self.sbs.stem(t) for t in word_tokenize(doc)]
