ARGS_COL_MAP = {
    'target': 'Target',
    'pretrained_file': 'Pretrained Model',
    'embed_dim': 'Embedding Dimension',
    'words_dim': 'Word Dimension',
    'epochs': 'Epochs',
    'output_channel': 'CNN Output Channels',
    'num_layers': 'LSTM Layers',
    'hidden_dim': 'LSTM Hidden Dimension',
    'l2logreg_c': 'BoW LR C',
    'lr': 'Learning Rate',
    'patience': 'Patience',
    'mode': 'Mode',
    'cuda_block': 'CUDA Blocking',
    'imbalance_fix': 'Class Imbalance Fix',
    'data_version': 'Data',
    'bidirectional': 'LSTM Bidirectional',
    'classifier': 'BoW Classifier',
    'max_tokens': 'Max Tokens'
}

CUSTOM_COLS = ['acc', 'bal_acc']

# The desired order of the columns, just for aesthetics
COLS_IN_ORDER = ['Run Name', 'Finished', 'Table', 'Platform', 'Model', 'Target',
                 'Data', 'Mode', 'Pretrained Model', 'Embedding Dimension', 'Word Dimension',
                 'BoW Classifier', 'BoW LR C', 'CNN Output Channels', 'LSTM Layers', 'LSTM Hidden Dimension', 'LSTM Bidirectional',
                 'Max Tokens',
                 'CUDA Blocking', 'Epochs', 'Learning Rate', 'Patience', 'Class Imbalance Fix',
                 'Accuracy', 'Balanced Accuracy', 'AUC', 'F1', 'Recall', 'Precision', 'Specificity',
                 'PPV', 'NPV', 'TP', 'TN', 'FP', 'FN', 'LaTeX String', 'Table Extra']
