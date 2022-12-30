from tqdm import tqdm
import torch
import time
import pandas as pd
from utils import print_from_history
from evaluators.calculate_metrics import add_epoch_perf
import os
from copy import deepcopy
from models.cnn.model import CNN


class NeuralTrainer(object):
    METRICS = ['acc', 'bal_acc', 'auc', 'prec', 'rec', 'f1', 'loss']  # Performance metrics to evaluate

    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 config):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = config.device
        self.patience = config.patience
        self.monitor_metric = config.monitor_metric
        self.best_moniter_metric = 0
        self.iters_not_improved = 0
        self.early_stop = None
        self.start = None
        self.n_epochs = config.epochs
        self.snapshot_path = os.path.join(config.results_dir_model, config.run_name + ".pt")
        self.model_file = config.model_file

    def fit(self,
            train_loader,
            dev_loader,
            test_loader):

        train_history = self.create_history_pd()
        dev_history = self.create_history_pd()
        test_history = self.create_history_pd()

        # Set Pandas setting to allow more columns to print out epoch results
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 100)

        # Convert generators to lists so they don't get consumed after one epoch
        train_loader = list(train_loader)
        dev_loader = list(dev_loader)
        test_loader = list(test_loader)
        self.start = time.time()

        for epoch in range(self.n_epochs):

            self.model.train()

            predicted_labels = torch.empty(0, device=self.device)
            target_labels = torch.empty(0, device=self.device).int()
            train_losses = torch.empty(0, device=self.device)

            for inputs, targets in tqdm(train_loader, leave=True, desc=f'Training epoch {epoch + 1}/{self.n_epochs}'):
                targets = targets.view(-1, 1).float()
                inputs = torch.transpose(inputs, 0, 1)

                # Move data to GPU
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # If needed, print current balance
                # print(f'For this batch, {targets.cpu().detach().numpy().sum()} / {
                # len(targets.cpu().detach().numpy())} are 1')

                # zero the gradient
                self.optimizer.zero_grad()

                # forward pass
                outputs = self.model(inputs)
                train_loss = self.loss_fn(outputs, targets)

                # Instead, keep track of predictions and labels so can do full metrics
                predictions = torch.sigmoid(outputs)
                predicted_labels = torch.cat((predicted_labels, predictions))
                target_labels = torch.cat((target_labels, targets.int()))

                # backward pass and optimize
                train_loss.backward()
                self.optimizer.step()

                # Store loss
                train_loss_tensor = torch.tensor([train_loss.item()], device=self.device)
                train_losses = torch.cat((train_losses, train_loss_tensor))

            mean_train_losses = torch.mean(train_losses)

            # Evaluate performance on training set and update
            train_history = add_epoch_perf(target_labels, predicted_labels, mean_train_losses, train_history)

            # Evaluate on dev set ----------------------------------------
            self.model.eval()
            with torch.no_grad():
                dev_losses = torch.empty(0, device=self.device)
                dev_predicted_labels = torch.empty(0, device=self.device)
                dev_target_labels = torch.empty(0, device=self.device).int()

                for inputs, targets in dev_loader:
                    targets = targets.view(-1, 1).float()
                    inputs = torch.transpose(inputs, 0, 1)
                    # Move data to GPU
                    dev_inputs, dev_targets = inputs.to(self.device), targets.to(self.device)

                    # forward pass
                    dev_outputs = self.model(dev_inputs)
                    dev_loss = self.loss_fn(dev_outputs, targets)

                    # get prediction and labels
                    dev_predictions = torch.sigmoid(dev_outputs)
                    dev_predicted_labels = torch.cat((dev_predicted_labels, dev_predictions))
                    dev_target_labels = torch.cat((dev_target_labels, dev_targets.int()))

                    # Store loss
                    dev_loss_tensor = torch.tensor([dev_loss.item()], device=self.device)
                    dev_losses = torch.cat((dev_losses, dev_loss_tensor))

                # Evaluate metrics based on prediction and labels
                mean_dev_losses = torch.mean(dev_losses)
                dev_history = add_epoch_perf(dev_target_labels, dev_predicted_labels, mean_dev_losses, dev_history)
                epoch_monitor_metric = dev_history.loc[dev_history.index[-1], self.monitor_metric]

                # Print Epoch Results so far
                print_from_history(dev_history, -1, self.start, epoch, self.n_epochs)

                # Update validation results
                if epoch_monitor_metric > self.best_moniter_metric:
                    self.iters_not_improved = 0
                    self.best_moniter_metric = epoch_monitor_metric
                    model_to_save = deepcopy(self.model)
                    torch.save({
                        'model_state_dict': model_to_save.state_dict(),
                        'config': self.config,
                        'loss_fn': self.loss_fn
                    }, self.snapshot_path)
                else:
                    self.iters_not_improved += 1
                    if self.iters_not_improved >= self.patience:
                        self.early_stop = True
                        print("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, self.best_moniter_metric))
                        break

        test_history = self.evaluate_on_test_split(model_to_save, test_loader, test_history)
        return train_history, dev_history, test_history, self.start

    def evaluate_on_test_split(self, model, test_loader, history):
        model.eval()

        with torch.no_grad():
            test_losses = torch.empty(0, device=self.device)
            test_predicted_labels = torch.empty(0, device=self.device)
            test_target_labels = torch.empty(0, device=self.device).int()

            for inputs, targets in test_loader:
                targets = targets.view(-1, 1).float()
                inputs = torch.transpose(inputs, 0, 1)
                # Move data to GPU
                test_inputs, test_targets = inputs.to(self.device), targets.to(self.device)

                # forward pass
                test_outputs = model(test_inputs)
                test_loss = self.loss_fn(test_outputs, targets)

                # get prediction and labels
                test_predictions = torch.sigmoid(test_outputs)
                test_predicted_labels = torch.cat((test_predicted_labels, test_predictions))
                test_target_labels = torch.cat((test_target_labels, test_targets.int()))

                # Store loss
                test_loss_tensor = torch.tensor([test_loss.item()], device=self.device)
                test_losses = torch.cat((test_losses, test_loss_tensor))

            # Evaluate metrics based on prediction and labels
            mean_test_losses = torch.mean(test_losses)
            history = add_epoch_perf(test_target_labels, test_predicted_labels, mean_test_losses, history)

            # Print Epoch Results so far
            print("Here are the results on our test split:")
            print_from_history(history, -1, self.start, 0, self.n_epochs)

            # Return test split history
            return history

    def eval_only(self,
                  test_loader):

        test_history = self.create_history_pd()

        # Set Pandas setting to allow more columns to print out epoch results
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 100)

        # Convert generators to lists so they don't get consumed after one epoch
        test_loader = list(test_loader)
        self.start = time.time()

        # Load model
        checkpoint = torch.load(self.model_file)
        loading_model = CNN(config=checkpoint['config'])
        loading_model.load_state_dict(checkpoint['model_state_dict'])
        loading_model.eval()
        loaded_model = loading_model.to(self.device)

        test_history = self.evaluate_on_test_split(loaded_model, test_loader, test_history)
        return test_history, self.start

    @staticmethod
    def create_history_pd():
        """
        Creates a blank history data frame, to keep track of performance metrics
        :return: history data frame with keys and blank arrays
        """

        # history = pd.DataFrame(columns=[self.METRICS])
        history = pd.DataFrame()

        return history
