import time
import pandas as pd
import numpy as np
import torch


def series_to_matrix(series):
    """
    Converts a pandas series of string repersented arrays to a numpy matrix.
    Used because of how we stored our BoW data

    :param series: A pandas series where each item is a string of an array e.g. '[1,2,3]'
    :return: a numpy matrix where each element from the series is a row
    """
    a_list = series.to_list()

    try:
        eval_list = [eval(x) for x in a_list]
    except TypeError:
        eval_list = a_list

    # print(eval_list[0])

    matrix = np.row_stack(eval_list)
    return matrix


def move_to_device(obj, device):
    # This function is from a guy online:
    # https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283/2

    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return res
    # elif isinstance(obj, int) or isinstance(obj, np.float64):
    #    return obj
    else:
        raise TypeError("Invalid type for move_to_device, supports tensor, list, dict"
                        f"The type is {type(obj)}, and the obj itself: {obj}")


def print_from_history(history, index, start_time, epoch, n_epochs):
    """
    Prints out the history in a nice format from the provided history at the given index
    Also takes in start_time to print out elapsed time in minutes, and the epoch we're at and total

    :param history: pd containing the results from running the model so far
    :param index: integer representing what from the history to print
    :param start_time: time training and evaluating started
    :param epoch:  current epoch
    :param n_epochs: how many epochs being used
    :return:
    """

    index = history.index[index]  # Re-cast this so that -1 index can be used with .loc below

    minutes_elapsed = int(round((time.time() - start_time) / 60, 0))
    epoch_header = ['Minutes', 'Epoch', 'Dev/Acc', 'Dev/BalAcc', 'Dev/AUC', 'Dev/Pr', 'Dev/Rec', 'Dev/F1',
                    'Dev/Loss']
    epoch_metrics = [[minutes_elapsed, f'{epoch}/{n_epochs}',
                      round(history.loc[index, "acc"], 3),
                      round(history.loc[index, "bal_acc"], 3),
                      round(history.loc[index, "auc"], 3),
                      round(history.loc[index, "prec"], 3),
                      round(history.loc[index, "rec"], 3),
                      round(history.loc[index, "f1"], 3),
                      round(history.loc[index, "loss"], 3)]]
    print(pd.DataFrame(epoch_metrics, columns=epoch_header, index=['']))
