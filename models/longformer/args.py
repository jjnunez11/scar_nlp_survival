import models.args
# layout as per pytorch-lightning best practices
# https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html


def get_args():
    parser = models.args.get_args()

    # Trainer arguments
    parser.add_argument('--load-ckpt', type=str, default='',
                        help='Provide path to a pytorch-lightning .ckpt file to load\n'
                             'a checkpoint. If string is empty, starts from scratch.\n'
                             'Loading a checkpoint will ignore other passed arguments and use'
                             'those from the checkpoint.')
    # Model arguments
    parser.add_argument('--lr', type=float, default=0.00005)  # thesis 0.00001 but usually set higher
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help="Maximum number of tokens to processes in, though Longformer can handle 4096")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--attention-window', type=int, default=512,
                        help="Size of attention window around each token.")

    parser.add_argument('--pretrained_dir',
                        default='')
    parser.add_argument('--pretrained-file', default='allenai/longformer-base-4096',
                        choices=['allenai/longformer-base-4096',
                                 'allenai/longformer-large-4096'])
    # Program arguments
    parser.add_argument('--count_tokens', default=False,
                        help="Instead of training a model, just count the number of tokens in dataset")


    args = parser.parse_args()
    return args
