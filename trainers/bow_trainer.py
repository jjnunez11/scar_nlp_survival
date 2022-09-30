import time
import pandas as pd
from utils import print_from_history, series_to_matrix
from copy import deepcopy
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
import numpy as np
from evaluators.calculate_metrics import add_epoch_perf
import os
import _pickle as cPickle
import bz2


class BoWTrainer(object):
    METRICS = ['acc', 'bal_acc', 'auc', 'prec', 'rec', 'f1', 'spec', 'loss']  # Performance metrics to evaluate

    def __init__(self,
                 config,
                 class_weight):
        self.classifier = config.classifier
        self.rf_estimators = config.rf_estimators
        self.elnet_l1_ratio = config.elnet_l1_ratio
        self.elnet_alpha = config.elnet_alpha
        self.elnet_power_t = config.elnet_power_t
        self.gbdt_estimators = config.gbdt_estimators
        self.gbdt_lr = config.gbdt_lr
        self.gbdt_max_depth = config.gbdt_max_depth
        self.l2logreg_c = config.l2logreg_c

        self.class_weight = class_weight
        self.best_dev_f1 = 0
        self.start = None

        # Need these to save model
        self.results_dir_model = config.results_dir_model
        self.run_name = config.run_name

    def fit(self,
            train_data,
            dev_data,
            n_epochs: int = 100):

        train_history = pd.DataFrame()
        dev_history = pd.DataFrame()

        # Set Pandas setting to allow more columns to print out epoch results
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 100)

        # Load the data
        train_x = series_to_matrix(deepcopy(train_data['vector']))
        train_y = deepcopy(train_data['label']).to_numpy()
        dev_x = series_to_matrix(deepcopy(dev_data['vector']))
        dev_y = deepcopy(dev_data['label']).to_numpy()

        self.start = time.time()

        warnings.warn('For our BoW models, epochs are simply re-training and evaluating our models each epoch')
        for epoch in range(n_epochs):

            # Create BoW Model
            if self.classifier == "rf":
                clf = RandomForestClassifier(n_estimators=self.rf_estimators,
                                             n_jobs=-1,
                                             class_weight=self.class_weight)
            elif self.classifier == "elnet":
                clf = SGDClassifier(loss='log', penalty='elasticnet',
                                    l1_ratio=self.elnet_l1_ratio,
                                    alpha=self.elnet_alpha,
                                    max_iter=1000,
                                    power_t=self.elnet_power_t,
                                    class_weight=self.class_weight)
            elif self.classifier == "gbdt":
                clf = GradientBoostingClassifier(n_estimators=self.gbdt_estimators,
                                                 learning_rate=self.gbdt_lr,
                                                 max_depth=self.gbdt_max_depth,
                                                 random_state=0)
            elif self.classifier == 'l2logreg':
                clf = LogisticRegression(penalty='l2',
                                         solver='lbfgs',
                                         max_iter=1000,
                                         C=self.l2logreg_c,
                                         class_weight=self.class_weight)

            # Fit model
            clf.fit(train_x, train_y)

            # Predict using train set
            predicted_labels = np.array(clf.predict_proba(train_x))
            predicted_labels = np.array([x[1] for x in predicted_labels])  # Get the probability for survival
            target_labels = np.array(train_y)

            train_loss = 0  # np.mean(train_loss)

            # Evaluate performance on training set and update
            train_history = add_epoch_perf(target_labels, predicted_labels, train_loss, train_history)

            # Evaluate on dev set ----------------------------------------
            predicted_labels = np.array(clf.predict_proba(dev_x))
            predicted_labels = np.array([x[1] for x in predicted_labels])  # Get the probability for survival
            target_labels = np.array(dev_y)
            dev_loss = 0  # np.mean(dev_loss)
            dev_history = add_epoch_perf(target_labels, predicted_labels, dev_loss, dev_history)

            # Print Epoch Results so far
            print_from_history(dev_history, -1, self.start, epoch, n_epochs)

            # Save model
            models_filename = os.path.join(self.results_dir_model, self.run_name + f"_e{epoch}") + '.pbz2'
            with bz2.BZ2File(models_filename, 'w') as f2:
                cPickle.dump(clf, f2)

        return train_history, dev_history, self.start

