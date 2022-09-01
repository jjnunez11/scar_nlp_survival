ARGS_COL_MAP = {
    'target': 'Target',
    'pretrained_file': 'Pretrained Model',
    'embed_dim': 'Embedding Dimension',
    'words_dim': 'Word Dimension',
    'epochs': 'Epochs',
    'l2logreg_c': 'BoW LR C',
    'rf_estimators': 'BoW RF Estimators',
    'lr': 'Learning Rate',
    'patience': 'Patience',
    'mode': 'Mode',
    'cuda_block': 'CUDA Blocking',
    'imbalance_fix': 'Class Imbalance Fix',
    'data_version': 'Data',
    'classifier': 'BoW Classifier',
    'max_tokens': 'Max Tokens',
    'dropout': 'Dropout',
    # CNN hyper-parameters
    'output_channel': 'CNN Output Channels',
    'weight_decay': 'CNN Weight Decay',
    # LSTM hyper-parameters
    'bidirectional': 'LSTM Bidirectional',
    'num_layers': 'LSTM Layers',
    'hidden_dim': 'LSTM Hidden Dimension',
    'embed_droprate': 'LSTM Embed Dropout',
    'wdrop': 'LSTM Wt Dropout'
}

CUSTOM_COLS = ['acc', 'bal_acc']

# The desired order of the columns, just for aesthetics
COLS_IN_ORDER = ['Run Name', 'Finished', 'Table', 'Platform', 'Model', 'Target',
                 'Data', 'Mode', 'Pretrained Model', 'Embedding Dimension', 'Word Dimension', 'Dropout',
                 'BoW Classifier', 'BoW LR C', 'BoW RF Estimators',
                 'CNN Output Channels', 'CNN Weight Decay',
                 'LSTM Layers', 'LSTM Hidden Dimension', 'LSTM Bidirectional', 'LSTM Wt Dropout', 'LSTM Embed Dropout',
                 'Max Tokens',
                 'CUDA Blocking', 'Epochs', 'Learning Rate', 'Patience', 'Class Imbalance Fix',
                 'Accuracy', 'Balanced Accuracy', 'AUC', 'F1', 'Recall', 'Precision', 'Specificity',
                 'PPV', 'NPV', 'TP', 'TN', 'FP', 'FN', 'LaTeX String', 'Table Extra']
