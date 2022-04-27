from transformers import LongformerTokenizer
from datasets.scar_transformer import SCARTransformer


class SCARLongformer(SCARTransformer):
    def __init__(self, config, undersample=False):
        super().__init__(config, LongformerTokenizer, undersample)
