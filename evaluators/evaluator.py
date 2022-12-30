from utils import print_from_history
import os
import pandas as pd
import datetime
from evaluators.evaluator_globals import ARGS_COL_MAP, COLS_IN_ORDER


class Evaluator(object):
    METRICS = ['acc', 'bal_acc', 'auc', 'prec', 'rec', 'f1', 'loss']  # Performance metrics to evaluate

    def __init__(self,
                 model_name,
                 history,
                 config,
                 start_time):
        self.history = history
        self.args = config
        self.n_epochs = config.epochs
        self.start = start_time
        self.end = datetime.datetime.now()
        self.model_name = model_name
        self.table = config.table
        self.table_extra = config.table_extra
        self.target = config.target
        self.run_name = config.run_name
        self.results_dir_target = config.results_dir_target
        self.results_dir_model = config.results_dir_model

        if self.args.debug:
            self.model_name = model_name + "_debug"
        else:
            self.model_name = model_name

        # If a results file has not been created for a target, make it
        self.results_for_target = os.path.join(self.results_dir_target, self.target + '_results.csv')

        if not os.path.exists(self.results_for_target):
            df = pd.DataFrame(columns=['Datetime'])
            df.to_csv(self.results_for_target)

    def get_best_epoch_metric(self, metric):
        """
        Returns the index position  of history with the highest value for the given metric

        :param metric: string of the metric e.g. auc
        :return: the index position  with the highest value for this metric
        """
        history = self.history
        best_epoch_metric_idx = history[metric].idxmax()

        # index = history.index[best_auc_epoch_idx]  # Convert this index position to index label

        return best_epoch_metric_idx

    def get_best_epoch_auc_idx(self):
        return self.get_best_epoch_metric('auc')

    def get_best_epoch_f1_idx(self):
        return self.get_best_epoch_metric('f1')

    def get_best_epoch_bal_acc_idx(self):
        return self.get_best_epoch_metric('bal_acc')

    def print_best_auc(self):
        history = self.history
        best_auc_epoch_idx = self.get_best_epoch_auc_idx()
        print_from_history(history, best_auc_epoch_idx, self.start, best_auc_epoch_idx, self.n_epochs)

    def print_best_bal_acc(self):
        history = self.history
        best_auc_epoch_idx = self.get_best_epoch_bal_acc_idx()
        print_from_history(history, best_auc_epoch_idx, self.start, best_auc_epoch_idx, self.n_epochs)

    def print_best_f1(self):
        history = self.history
        best_f1_epoch_idx = self.get_best_epoch_f1_idx()
        print_from_history(history, best_f1_epoch_idx, self.start, best_f1_epoch_idx, self.n_epochs)

    def append_to_results(self):
        """
        Write a selection of the arguments to a general results file shared for
        all models and runs for a specific target

        :return: Nothing, but writes to the above.
        """

        args_dict = vars(self.args)

        # Create dictionary to add to pandas dataframe, and add the datetime of evaluation, that Platform being
        # this repo, and the name of the model being used
        to_append = {'Run Name': self.run_name,
                     'Finished': self.end.strftime("%Y%m%d-%H%M"),
                     'Platform': 'SCAR_NLP',
                     'Table': self.table,
                     'Model': self.model_name}

        # Add in the accuracy, balanced accuracy, F1 and AUC from the epoch with the best balanced accuracy
        best_bal_acc_idx = self.get_best_epoch_bal_acc_idx()
        best_bal_acc_index = self.history.index[best_bal_acc_idx]
        best_bal_acc_epoch = self.history.loc[best_bal_acc_index, :]
        best_bal_acc = best_bal_acc_epoch['bal_acc']
        best_acc = best_bal_acc_epoch['acc']
        best_f1 = best_bal_acc_epoch['f1']
        best_auc = best_bal_acc_epoch['auc']

        to_append['Balanced Accuracy'] = best_bal_acc
        to_append['Accuracy'] = best_acc
        to_append['AUC'] = best_auc
        to_append['F1'] = best_f1

        # Add in additional metrics
        to_append['Recall'] = best_bal_acc_epoch['rec']
        to_append['Specificity'] = best_bal_acc_epoch['spec']
        to_append['Precision'] = best_bal_acc_epoch['prec']
        to_append['Loss'] = best_bal_acc_epoch['loss']

        # Also add in confusion matrix stuff
        to_append['PPV'] = best_bal_acc_epoch['ppv']
        to_append['NPV'] = best_bal_acc_epoch['npv']
        to_append['TP'] = best_bal_acc_epoch['tp']
        to_append['TN'] = best_bal_acc_epoch['tn']
        to_append['FP'] = best_bal_acc_epoch['fp']
        to_append['FN'] = best_bal_acc_epoch['fn']

        # Put in a string with results formatted for LaTeX
        # Some tables need an extra column, include that if it exists
        decimals = 3
        if self.table_extra == '':
            latex_string = " & ".join([self.model_name,
                                       str(round(best_bal_acc_epoch['acc'], decimals)),
                                       str(round(best_bal_acc_epoch['bal_acc'], decimals)),
                                       str(round(best_bal_acc_epoch['auc'], decimals)),
                                       str(round(best_bal_acc_epoch['f1'], decimals)),
                                       str(round(best_bal_acc_epoch['prec'], decimals)),
                                       str(round(best_bal_acc_epoch['rec'], decimals))])
        else:
            latex_string = " & ".join([self.model_name,
                                       self.table_extra,
                                       str(round(best_bal_acc_epoch['acc'], decimals)),
                                       str(round(best_bal_acc_epoch['bal_acc'], decimals)),
                                       str(round(best_bal_acc_epoch['auc'], decimals)),
                                       str(round(best_bal_acc_epoch['f1'], decimals)),
                                       str(round(best_bal_acc_epoch['prec'], decimals)),
                                       str(round(best_bal_acc_epoch['spec'], decimals))])

        latex_string = latex_string + r" \\"
        print(latex_string)
        to_append['LaTeX String'] = latex_string
        to_append['Table Extra'] = self.table_extra

        # From our arguments, add the selected arguments to the dictionary to add,
        # but change the names according to the map
        for key in ARGS_COL_MAP:
            col_header = ARGS_COL_MAP[key]
            if key in args_dict:
                # If this key is in our arugments, append the value so it's added
                to_append[col_header] = args_dict[key]
            else:
                # Depending on the model, some args will be missing, indicate with a blank
                to_append[col_header] = ''

        # Read in the prior results df, and append this result via the dict
        df = pd.read_csv(self.results_for_target)
        df = df.append(to_append, ignore_index=True)
        # Set index, and reorder the columns
        df = df.set_index("Datetime")
        df = df[COLS_IN_ORDER]

        df.to_csv(self.results_for_target)

    def write_result_history(self):
        """
        Prints the history from this run

        :return: Nothing, but prints out history to a file
        """

        # datetime_on_finish = self.end.strftime("%Y%m%d-%H%M")
        filename = os.path.join(self.results_dir_model, self.run_name + '.csv')

        # Save the history of this run to a csv
        self.history.to_csv(filename)

        # Then open the csv, and print out all of the argparser arguments
        f = open(filename, "a")
        f.write('\nRun Arguments:\n')

        args_dict = vars(self.args)
        for key in args_dict.keys():
            f.write(f'{key}: {args_dict[key]}\n')

        f.close()
