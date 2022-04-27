from models.bert.args import get_args
from models.bert.model import BERT
from datasets.scar_bert import SCARBERT
from models.transformer_main import transformer_main

if __name__ == '__main__':
    transformer_main("BERT", BERT, SCARBERT, get_args())
