import models.args
import os

def get_args():
    parser = models.args.get_args()
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help="Maximum number of tokens to use with Big Bird, max 4096")
    parser.add_argument('--gradient-checkpointing', type=bool, default=False)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0)

    parser.add_argument('--block-size', type=int, default=64,
                        help="BigBird block size, defaults to 64 per their paper")
    parser.add_argument('--num-blocks', type=int, default=3,
                        help="BigBird number of random blocks, defaults 3 as this is their"
                             "suggested default, but might want to try 0 for their ETC model.")
    parser.add_argument('--attention-type', type=str, default="block_sparse",
                        help="Must be either block_sparse or original_full. However,"
                             "original_full can only go up to 1024 tokens, so use default.")

    # Directory to import in pretrained BERT model
    ## parser.add_argument('--pretrained_dir',
    ##                    default=os.path.join(r"C:\Users\jjnunez\PycharmProjects", 'hedwig-data', 'models'))
    ##parser.add_argument('--pretrained_file', default='bert-base-uncased',
    ##                    choices=['bert-base-uncased',
    ##                             'bert-large-uncased',
    ##                             'bert_pretrain_output_all_notes_150000',
    ##                             'bert_pretrain_output_disch_100000',
    ##                             'biobert_pretrain_output_all_notes_150000',
    ##                             'biobert_pretrain_output_disch_100000'])

    args = parser.parse_args()
    return args
