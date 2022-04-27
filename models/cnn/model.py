import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext

"""
From the Kim CNN model.py from the Hedwig repository
"""


class CNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        output_channel = config.output_channel
        target_classes = config.target_classes
        vocab_size = config.vocab_size
        words_dim = config.words_dim
        self.mode = config.mode
        self.device = config.device  # Device, either a CUDA gpu or CPU
        ks = 3  # There are three conv nets here

        input_channel = 1
        # Setup Embeddings, whether Pre-trained and fixed
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(config.vocab_size, config.words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            glove = torchtext.vocab.GloVe(
                dim=self.D)  # name="6B", max_vectors=10000 TODO: Options for different vectors
            self.embed = nn.Embedding.from_pretrained(glove.vectors, freeze=True)
        elif config.mode == 'non-static':
            glove = torchtext.vocab.GloVe(
                dim=self.D)  # name="6B", max_vectors=10000 TODO: Options for different vectors
            self.embed = nn.Embedding.from_pretrained(glove.vectors, freeze=False)
        else:
            print("Unsupported Mode")
            exit()

        print(f'self.embed num_embeddings is: {self.embed.num_embeddings} '
              f'while embedding_dim is {self.embed.embedding_dim}')

        #####
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(vocab_size, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        elif config.mode == 'multichannel':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
            input_channel = 2
        else:
            print("Unsupported Mode")
            exit()
        #####

        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, words_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, words_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, words_dim), padding=(4, 0))

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(ks * output_channel, target_classes)

        # Send model to device from init
        self.to(self.device)

    def forward(self, x):
        if self.mode == 'rand':
            word_input = self.embed(x)  # (batch, sent_len, embed_dim)
            x = word_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'static':
            static_input = self.static_embed(x)
            x = static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'non-static':
            non_static_input = self.non_static_embed(x)
            x = non_static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.mode == 'multichannel':
            non_static_input = self.non_static_embed(x)
            static_input = self.static_embed(x)
            x = torch.stack([non_static_input, static_input], dim=1)  # (batch, channel_input=2, sent_len, embed_dim)
        else:
            print("Unsupported Mode")
            exit()

        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        # (batch, channel_output) * ks
        x = torch.cat(x, 1)  # (batch, channel_output * ks)
        x = self.dropout(x)
        logit = self.fc1(x)  # (batch, target_size)
        return logit
