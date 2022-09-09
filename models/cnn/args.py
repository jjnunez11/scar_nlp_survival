import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--mode', type=str, default='rand',
                        choices=['rand', 'static', 'non-static', 'multichannel'])
    parser.add_argument('--output-channel', type=int, default=500)
    parser.add_argument('--words-dim', type=int, default=300)
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.8)  # thesis default 0.5
    parser.add_argument('--lr', type=float, default=0.00005)  # thesis default 0.00001
    # TODO_ parser.add_argument('--epoch-decay', type=int, default=15)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--pretrained-dir', default=os.path.join(r"C:\Users\jjnunez\PycharmProjects", 'hedwig-data', 'embeddings', 'word2vec'))
    parser.add_argument('--pretrained-file', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--batch-size', type=int, default=16)
    # TODO_ parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'char_cnn'))
    # TODO_ parser.add_argument('--resume-snapshot', type=str)
    # TODO_ parser.add_argument('--trained-model', type=str)

    args = parser.parse_args()
    return args
