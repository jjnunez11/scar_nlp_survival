import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="PyTorch deep learning models for document classification")

    parser.add_argument('--no-cuda', action='store_false', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cuda_block', default=False, dest='cuda_block', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--monitor_metric', type=str, default="bal_acc", choices=['bal_acc', 'f1', 'auc'])
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--imbalance-fix', default='loss_weight', choices=['loss_weight', 'undersampling', 'none'],
                        help='How to deal with the class imbalance in our dataset')
    parser.add_argument('--data-dir',
                        default=os.path.join(r'C:\Users\jjnunez\PycharmProjects', 'scar_nlp_data', 'data'))
    parser.add_argument('--results-dir',
                        default=os.path.join(r'C:\Users\jjnunez\PycharmProjects\scar_nlp_survival', 'results'))
    parser.add_argument('--data-version', type=str, default='ppv4',
                        help='Version of preprocessing used for this dataset')
    parser.add_argument('--target', type=str, help='The specific target for the prediction')
    parser.add_argument('--table', type=str, default='',
                        help="Allow a string to be passed to label results for which tables they'll be part of")
    parser.add_argument('--table_extra', type=str, default='',
                        help="Extra string to put as a column in the table, e.g. to show how many needs.")
    parser.add_argument('--dataset', type=str, default='SCAR', choices=['SCAR'])
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help="If included, loads a smaller training set to save time while debugging")
    parser.add_argument('--eval_only', default=False, dest='eval_only', action='store_true',
                        help="If provided, will not train a model, but will instead load a model and evaluate that")
    parser.add_argument('--model-file',
                        default=None,
                        help="Absolute path to a PyTorch model checkpoint (neural models) or pickle dump (BoW)"
                             " to load and evaluate")

    return parser
