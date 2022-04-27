# Inspired from https://huggingface.co/google/bigbird-roberta-large
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BigBirdModel, AdamW, get_linear_schedule_with_warmup
from torchmetrics.functional import accuracy, precision, recall, auroc, f1
from torchmetrics import Accuracy
import torch


class BigBird(pl.LightningModule):
    # Set up the classifier
    def __init__(self, config, loss_fn, steps_per_epoch):
        super().__init__()

        ## pretrained_model_path = os.path.join(config.pretrained_dir, config.pretrained_file)
        self.bigbird = BigBirdModel.from_pretrained("google/bigbird-roberta-base",
                                                    block_size=config.block_size,
                                                    num_random_blocks=config.num_blocks,
                                                    attention_type=config.attention_type,
                                                    max_position_embeddings=config.max_tokens,
                                                    use_cache=False,
                                                    gradient_checkpointing=config.gradient_checkpointing
                                                    # hidden_size=512,
                                                    # num_hidden_layers=4,
                                                    # num_attention_heads=4,
                                                    # intermediate_size=1024
                                                    )
        self.classifier = nn.Linear(self.bigbird.config.hidden_size, out_features=1)  # Change if multi-label
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = config.epochs
        self.lr = config.lr
        self.loss_fn = loss_fn
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
        output = self.bigbird(input_ids=input_ids, attention_mask=attn_mask)
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
        dev_epoch_metrics = self.calculate_metrics(pred_labels, target_labels_int)
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

        if len(output) > 16:
            # Calculate and then log the metrics being used for this proejct
            dev_epoch_metrics = self.calculate_metrics(pred_labels, target_labels_int)
            self.log('dev_perf',
                     dev_epoch_metrics,
                     prog_bar=False,
                     logger=True,
                     on_epoch=True)

            # Keep balanced accuracy a full on torchmetrics module, as part of this class, for early stopping
            bal = self.val_bal(pred_labels, target_labels_int)
            self.log('val_bal', bal, prog_bar=True, logger=False)

            print(f'\nValidation epoch completed, used {len(pred_labels)} examples')
        else:
            print(f'\nValidation sanity check completed, not storing performance')

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
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

    @staticmethod
    def calculate_metrics(pred_labels, target_labels):
        """
        This function should be replaceable with MetricCollection from torch metrics,
        but it seems its not quite ready yet with respect to pl integration
        So do things manually

        :param pred_labels: tensor representing the predicted labels from a epoch
        :param target_labels: tensor representing the target labels from a epoch
        :return: dict of the metrics with their values
        """

        epoch_metrics = {'acc': accuracy(pred_labels, target_labels),
                         'bal_acc': accuracy(pred_labels, target_labels,
                                             average='macro',
                                             num_classes=2,
                                             multiclass=True),
                         'auc': auroc(pred_labels, target_labels,
                                      pos_label=1),
                         'prec': precision(pred_labels, target_labels),
                         'rec': recall(pred_labels, target_labels),
                         'f1': f1(pred_labels, target_labels),
                         'loss': 0
                         }

        return epoch_metrics
