from models.longformer.args import get_args
from models.longformer.model import Longformer
from datasets.scar_longformer import SCARLongformer
from models.transformer_main import transformer_main

if __name__ == '__main__':
    transformer_main("Longformer", Longformer, SCARLongformer, get_args())
