import models.args
import os

def get_args():
    parser = models.args.get_args()
    parser.add_argument('--max-tokens', type=int, default=512,
                        help="Maximum number of tokens to use with BERT, max 512")

    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=12)

    # Directory to import in pretrained BERT model
    parser.add_argument('--pretrained_dir',
                        default=os.path.join(r"C:\Users\jjnunez\PycharmProjects", 'hedwig-data', 'models'))
    parser.add_argument('--pretrained_file', default='bert-base-uncased',
                        choices=['bert-base-uncased',
                                 'bert-large-uncased',
                                 'bert_pretrain_output_all_notes_150000',
                                 'bert_pretrain_output_disch_100000',
                                 'biobert_pretrain_output_all_notes_150000',
                                 'biobert_pretrain_output_disch_100000'])

    # Program Arguments
    parser.add_argument('--count_tokens', default=False,
                        help="Instead of training a model, just count the number of tokens in dataset")

    args = parser.parse_args()
    return args
