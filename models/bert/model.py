# Inspired from https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b
# and from: https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import os
from torchmetrics import Accuracy
import torch
from evaluators.calculate_metrics import calculate_metrics


class BERT(pl.LightningModule):
    # Set up the classifier
    def __init__(self, config, loss_fn, steps_per_epoch):
        super().__init__()
        self.save_hyperparameters()
        pretrained_model_path = os.path.join(config.pretrained_dir, config.pretrained_file)
        self.bert = BertModel.from_pretrained(pretrained_model_path, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, out_features=1)  # Change if multi-label
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = config.epochs
        self.lr = config.lr
        self.loss_fn = loss_fn
        self.weight_decay = config.weight_decay
        # self.device = config.device  # pl seems to default send to cuda:0, using this for metrics

        # Initialize the metrics

        # Balanced Accuracy
        self.tr_bal = Accuracy(average='macro',
                               num_classes=2,
                               multiclass=True)
        self.val_bal = Accuracy(average='macro',
                                num_classes=2,
                                multiclass=True)

    def forward(self, input_ids, attn_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)
        sigmoided_output = torch.sigmoid(output)  # Loss function usually will include a sigmoid layer

        loss = 0
        if labels is not None:
            loss = self.loss_fn(output, labels)

        return loss, sigmoided_output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_labels = batch['label']

        loss, pred_labels = self(input_ids, attention_mask, target_labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": pred_labels.detach(), "labels": target_labels}

    def training_epoch_end(self, output):
        # Take all the target and predicted labels from the epoch and flatten into 1-dim tensors,
        # then make target into int
        pred_labels, target_labels = self.flatten_epoch_output(output)
        target_labels_int = target_labels.int()

        # Calculate and then log the metrics being used for this proejct
        dev_epoch_metrics = calculate_metrics(pred_labels, target_labels_int)
        self.log('train_perf',
                 dev_epoch_metrics,
                 prog_bar=False,
                 logger=True,
                 on_epoch=True)

        # Keep balanced accuracy a full on torchmetrics module, as part of this class, for progress bar
        bal = self.tr_bal(pred_labels, target_labels_int)
        self.log('tr_bal', bal, prog_bar=True, logger=False)

        print(f'\nTraining epoch completed, used {len(pred_labels)} examples')

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_labels = batch['label']

        loss, pred_labels = self(input_ids, attention_mask, target_labels)
        self.log("val_loss", loss, prog_bar=False, logger=True, on_epoch=True)

        return {"loss": loss, "predictions": pred_labels.detach(), "labels": target_labels}

    def validation_epoch_end(self, output):
        # Take all the target and predicted labels from the epoch and flatten into 1-dim tensors,
        # then make target into int
        try:
            pred_labels, target_labels = self.flatten_epoch_output(output)
        except:
            print(f'This is output len: {len(output)}')
            print(f'Error, this is output: {output}')
        target_labels_int = target_labels.int()

        # print(f'Here is the len of output: {len(output)}')
        # print(f'Here is the len of stacked_pred: {stacked_pred.size()}')

        # pred_labels = torch.zeros(pred_labels.size(), device=self.device) for testing metrics are working

        # Calculate and then log the metrics being used for this proejct
        dev_epoch_metrics = calculate_metrics(pred_labels, target_labels_int)
        self.log('dev_perf',
                 dev_epoch_metrics,
                 prog_bar=False,
                 logger=True,
                 on_epoch=True)

        # Keep balanced accuracy a full on torchmetrics module, as part of this class, for early stopping
        bal = self.val_bal(pred_labels, target_labels_int)
        self.log('val_bal', bal, prog_bar=True, logger=False)

        print(f'\nValidation epoch completed, used {len(pred_labels)} examples')

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        # Overwrites progress bar to remove v_num as I'm not using this and it annoys me
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    @staticmethod
    def flatten_epoch_output(output):
        """
        Helper function that takes all of the batches in an epoch_end and flattens it
        into one tensor each for the predictions and labels

        :param output: output handed via a epoch_end hook
        :param device: cuda device for the tensors
        :return: a tensor with all the predicted in this epoch, and one for all the target labels
        """
        try:
            epoch_pred_labels = torch.cat([x['predictions'] for x in output]).squeeze()
        except RuntimeError:
            print(f'Could not cat, here is the first prediction output: {output[0]["predictions"]}')
            # print(f'Could not cat, here is the batch tensor: {batch_pred_labels}')
            # print(f'Could not cat, here is the batch target tensor: {batch_target_labels}')

        try:
            epoch_target_labels = torch.cat([x['labels'] for x in output]).squeeze()
        except RuntimeError:
            print(f'Could not cat, here is the first prediction output: {output[0]["labels"]}')

        # first_run = True
        #
        # for batch in output:
        #     batch_pred_labels = batch['predictions'].squeeze()
        #     batch_target_labels = batch['labels'].squeeze()
        #
        #     if first_run:
        #         epoch_pred_labels = batch_pred_labels
        #         epoch_target_labels = batch_target_labels
        #         first_run = False
        #     else:
        #         try:
        #             epoch_pred_labels = torch.cat([epoch_pred_labels, batch_pred_labels])
        #         except RuntimeError:
        #             print(f'Could not cat, here is the batch tensor: {batch_pred_labels}')
        #             print(f'Could not cat, here is the epoch tensor: {epoch_pred_labels}')
        #             print(f'Could not cat, here is the batch target tensor: {batch_target_labels}')
        #             print(f'Could not cat, here is the epoch target tensor: {epoch_target_labels}')
        #         epoch_target_labels = torch.cat([epoch_target_labels, batch_target_labels])

        return epoch_pred_labels, epoch_target_labels

