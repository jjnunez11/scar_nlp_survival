import torch
import torch.nn as nn
import torchtext
# import torch.nn.functional as F
from models.lstm.embed_regularize import embedded_dropout
from models.lstm.weight_drop import WeightDrop


# Define the model
class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        # self.V = n_vocab
        self.D = config.words_dim    # Vector dimensions
        self.M = config.hidden_dim   # Hidden layer dimension
        self.K = 1                   # Not doing multi-class for now
        self.L = config.num_layers   # Number of layers
        self.device = config.device  # Device, either a CUDA gpu or CPU
        self.is_bidirectional = config.bidirectional
        self.mode = config.mode
        self.embed_droprate = config.embed_droprate  # Embedding droprate
        self.wdrop = config.wdrop

        # Setup Embeddings, whether Pre-trained and fixed
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(config.vocab_size, config.words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            glove = torchtext.vocab.GloVe(
                dim=self.D)  # name="6B", max_vectors=10000 TODO: Options for different vectors
            self.embed = nn.Embedding.from_pretrained(glove.vectors, freeze=True)
        elif self.mode == 'non-static':
            glove = torchtext.vocab.GloVe(
                dim=self.D)  # name="6B", max_vectors=10000 TODO: Options for different vectors
            self.embed = nn.Embedding.from_pretrained(glove.vectors, freeze=False)
        else:
            print("Unsupported Mode")
            exit()

        print(f'self.embed num_embeddings is: {self.embed.num_embeddings} '
              f'while embedding_dim is {self.embed.embedding_dim}')

        # Create model
        self.lstm = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=self.is_bidirectional)

        if self.wdrop > 0:
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=self.wdrop)
        self.dropout = nn.Dropout(config.dropout)

        if self.is_bidirectional:
            self.fc1 = nn.Linear(2 * self.M, self.K)
        else:
            self.fc1 = nn.Linear(self.M, self.K)

        self.dropout = nn.Dropout(config.dropout)

        # Send model to device from init
        self.to(self.device)

    def forward(self, x):
        if self.mode == 'rand':
            out = embedded_dropout(self.embed, x,
                                    dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.embed(
                    x)
        elif self.mode == 'static':
            out = embedded_dropout(self.static_embed, x,
                                    dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.static_embed(
                    x)
        elif self.mode == 'non-static':
            out = embedded_dropout(self.non_static_embed, x,
                                   dropout=self.embed_droprate if self.training else 0) \
                if self.embed_droprate else self.non_static_embed(x)
        else:
            print("Unsupported Mode")
            exit()

        out, _ = self.lstm(out)

        # max pool
        out, _ = torch.max(out, 1)

        # Dropout
        # Note: will get a warning saying that dropout will not work, but it is triggered
        # above on model creation, we are using dropout down here after the max-pooling layer,
        # so it does have an effect
        out = self.dropout(out)

        # we only want h(T) at the final time step
        out = self.fc1(out)
        return out
