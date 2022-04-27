from transformers import BertTokenizer
from datasets.scar_transformer import SCARTransformer


class SCARBERT(SCARTransformer):
    def __init__(self, config, undersample=False):
        super().__init__(config, BertTokenizer, undersample)
