import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--max-tokens', type=int, default=5000, help='Number of features to use in our BoW Model')
    parser.add_argument('--use-idf', type=bool, default=True,
                        help='True, uses term frequency and inverse document freq to adjust vectors, False just tf')
    parser.add_argument('--classifier', default='l2logreg', choices=['rf', 'elnet', 'gbdt', 'l2logreg'],
                        help='Non-neural machine learning model to use for the BoW classiifer ')

    # BoW ML Model Hyperparameters
    parser.add_argument('--rf-estimators', default=50, type=int)
    parser.add_argument('--elnet-l1-ratio', default=0.67, type=float)
    parser.add_argument('--elnet-alpha', default=0.1, type=float)
    parser.add_argument('--elnet-power-t', default=0.01, type=float)
    parser.add_argument('--gbdt-estimators', default=100, type=int)
    parser.add_argument('--gbdt-lr', default=0.1, type=float)
    parser.add_argument('--gbdt-max-depth', default=3, type=int)
    parser.add_argument('--l2logreg-c', default=1, type=float)  # thesis default 0.2

    # Extra arguments for saving models etc, can implement later
    # TODO_ parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'reg_lstm'))
    # TODO_ parser.add_argument('--resume-snapshot', type=str)
    # TODO_ parser.add_argument('--trained-model', type=str)

    args = parser.parse_args()
    return args
