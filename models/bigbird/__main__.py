import os
from copy import deepcopy
import torch.nn as nn
import torch
from evaluators.evaluator import Evaluator
from models.bigbird.args import get_args
from models.bigbird.model import BigBird
from trainers.bert_trainer import BERTTrainer, MyLogger
from datasets.scar_bigbird import SCAR_BigBird
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import datetime


if __name__ == '__main__':
    print("Training and evaluating a BigBird model")
    model_name = "BigBird"
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    args = get_args()

    # Set Device
    if args.cuda:
        args.device = torch.device('cuda:0')
        print("Using a CUDA GPU, woot!")
    else:
        args.device = 'cpu'
        print("Using a CPU, sad!")

    config = deepcopy(args)
    config.run_name = model_name + "_" + start_time

    scar_bigbird = SCAR_BigBird(config)

    # Set CUDA Blocking if needed, used when getting CUDA errors:
    if args.cuda_block:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    else:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

    # Loss and optimizer
    if args.imbalance_fix == 'loss_weight':
        target_perc = scar_bigbird.get_class_balance()  # Percentage of targets = 1
        pos_weight = (1 - target_perc) / target_perc
        print(f"Weighting our Loss Function to Balance Target Classes\n"
              f"Training examples with target=1 will get a factor of: {round(pos_weight, 3)}")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    elif args.imbalance_fix == 'none':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise Exception("Invalid method to fix the class imbalance provided, or not yet implemented")

    # Instantiate our Model
    steps_per_epoch = scar_bigbird.get_n_training()/config.batch_size
    model = BigBird(config, loss_fn, steps_per_epoch)

    # Save Checkpoint
    # saves a file like: input/BigBird-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # monitored quantity
        filename='BigBird-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,  # save the top 3 models
        mode='min',  # mode of the monitored quantity  for optimization
    )

    # Make and create if needed dir for loggers to log to
    results_dir_target = os.path.join(args.results_dir, config.target)  # dir for a targets results
    results_dir_model = os.path.join(results_dir_target, model_name)  # subdir for each model

    # Try out some different loggers
    # comet_logger = pl_loggers.CometLogger(save_dir=r"C:\Users\jjnunez\PycharmProjects\scar_nlp\comet_logs")
    # csv_logger = pl_loggers.CSVLogger(save_dir="csv_logs/")
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=results_dir_model)
    my_logger = MyLogger()

    trainer = BERTTrainer(config=config,
                          checkpoint_callback=checkpoint_callback,
                          logger=[my_logger, tb_logger]
    )


    # quit("Instantiated a trainer")

    # trainer = Trainer(model, optimizer, loss_fn, args.device, args.patience)

    trainer.fit(model, scar_bigbird)

    train_history, dev_history, start_time = trainer.get_results()

    evaluator = Evaluator(model_name, dev_history, args, start_time)

    # Use evaluator to print the best epochs
    print('\nBest epoch for AUC:')
    evaluator.print_best_auc()

    print('\nBest epoch for F1:')
    evaluator.print_best_f1()

    # Write the run history, and update the master results file
    evaluator.write_result_history()
    evaluator.append_to_results()

    quit("All done!")



