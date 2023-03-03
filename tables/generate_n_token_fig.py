"""
Script to determine the number of 

"""
from tables.table_globals import RESULT_TABLES_DIR
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_n_token_fig(model_name, config, dataset):
    f_stem = f'{model_name}_{config.target}_'
    f_fig = os.path.join(RESULT_TABLES_DIR, f_stem + "n_token.png")
    f_stats = os.path.join(RESULT_TABLES_DIR, f_stem + "n_token.txt")

    train_data = dataset.train_dataset
    n_consults = len(train_data)
    token_counts = np.empty(n_consults)

    for i in tqdm(range(n_consults)):
        doc_tensor = train_data[i]
        token_tensor = doc_tensor['input_ids']
        token_array = token_tensor.numpy()
        token_array = np.trim_zeros(token_array, 'b')
        # token_counts[i,:] = token_array
        n_tokens = len(token_array)
        token_counts[i] = n_tokens

    # Write some summary statistics
    f_stats = open(f_stats, 'w')
    f_stats.write(f"Summary Statistics for the Token Count when Training "
                  f"Documents are Tokenized using {model_name} Tokenizer \n")
    f_stats.write(f'Mean is {token_counts.mean()}\n')
    f_stats.write(f'Median is {np.median(token_counts)}\n')
    n_eql_512 = (token_counts <= 512).sum()
    perc_eql_512 = n_eql_512/n_consults
    f_stats.write(f'Number >= 512 {n_eql_512}\n')
    f_stats.write(f'Percentage >= 512 {perc_eql_512}\n')

    # Make Histogram
    _ = plt.hist(token_counts, bins=30)  # arguments are passed to np.histogram
    plt.axvline(512, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('Tokens in Document')
    plt.ylabel('Number of Documents')
    # plt.title("Histogram with 'auto' bins")
    plt.savefig(f_fig)
    plt.show()

    f_stats.close()
