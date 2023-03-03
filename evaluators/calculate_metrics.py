from torchmetrics.functional import accuracy, precision, recall, auroc, f1, specificity
from torchmetrics import ConfusionMatrix
import torch


def calculate_metrics(pred_labels, target_labels, loss=torch.tensor([0])):
    """
    This function should be replaceable with MetricCollection from torch metrics,
    but it seems its not quite ready yet with respect to pl integration
    So do things manually

    :param loss: loss, for when a model supports it, otherwise default to 0.
    :param pred_labels: tensor or ndarray representing the predicted labels from a epoch
    :param target_labels: tensor or ndarray representing the target labels from a epoch
    :return: dict of the metrics with their values
    """

    # Make a confusion matrix to calculate ppv/npv from
    confmat = ConfusionMatrix(num_classes=2)

    # Some inputs will be GPU tensors, while others are numpy arrays on CPU
    try:
        cm = confmat(pred_labels.cpu(), target_labels.cpu())
    except AttributeError:
        pred_labels = torch.from_numpy(pred_labels)
        target_labels = torch.from_numpy(target_labels).int()
        cm = confmat(pred_labels, target_labels)

    tn = cm[0, 0].item()
    tp = cm[1, 1].item()
    fn = cm[1, 0].item()
    fp = cm[0, 1].item()
    try:
        ppv = tp / (tp + fp)
    except ZeroDivisionError:
        ppv = 0
    try:
        npv = tn / (tn + fn)
    except ZeroDivisionError:
        npv = 0

    # Loss may or may not be a tensor object
    try:
        loss = loss.item()
    except AttributeError:
        pass

    # Calculate and make a dict with all the metrics
    epoch_metrics = {'acc': accuracy(pred_labels, target_labels).item(),
                     'bal_acc': accuracy(pred_labels, target_labels,
                                         average='macro',
                                         num_classes=2,
                                         multiclass=True).item(),
                     'auc': auroc(pred_labels, target_labels,
                                  pos_label=1).item(),
                     'prec': precision(pred_labels, target_labels).item(),
                     'rec': recall(pred_labels, target_labels).item(),
                     'spec': specificity(pred_labels, target_labels).item(),
                     'f1': f1(pred_labels, target_labels).item(),
                     'loss': loss,
                     'ppv': ppv,
                     'npv': npv,
                     'tp': tp,
                     'fp': fp,
                     'tn': tn,
                     'fn': fn
                     }

    return epoch_metrics


def add_epoch_perf(target_labels, pred_labels, loss, history):
    """
    Evaluate the various metrics for the supplied predicted and laballed targets, and append
    to the history dataframe
    """
    to_append = calculate_metrics(pred_labels, target_labels, loss)

    history = history.append(to_append, ignore_index=True)
    history.index.name = "epoch"  # only actually needs to be done once, but will just call again

    return history
