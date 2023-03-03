import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--mode', type=str, default='rand', choices=['rand', 'static', 'non-static'])
    parser.add_argument('--words-dim', type=int, default=300)
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.2)  # Thesis was 0.5
    parser.add_argument('--lr', type=float, default=0.00005)  # Thesis was 0.00001
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--embed-droprate', type=float, default=0.2, help="embedding dropout")  # thesis was 0.1
    parser.add_argument('--wdrop', type=float, default=0.3, help="weight drop")  # thesis was 0.3

    # Extra arguments from Hedwig that we can implement if desired
    # TODO_ parser.add_argument('--bottleneck-layer', action='store_true')
    # TODO_ parser.add_argument('--epoch-decay', type=int, default=15)
    # TODO_
    # TODO_ parser.add_argument('--beta-ema', type=float, default=0, help="temporal averaging")
    # TODO_ parser.add_argument('--tar', type=float, default=0.0, help="temporal activation regularization")
    # TODO_ parser.add_argument('--ar', type=float, default=0.0, help="activation regularization")

    parser.add_argument('--pretrained-dir', default=os.path.join(r"C:\Users\jjnunez\PycharmProjects", 'hedwig-data', 'embeddings', 'word2vec'))
    parser.add_argument('--pretrained_file', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--results-path', type=str, default=os.path.join('results', 'reg_lstm'))

    args = parser.parse_args()
    return args
