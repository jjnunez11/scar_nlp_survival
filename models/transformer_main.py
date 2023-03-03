import os
import sys
from copy import deepcopy
import torch.nn as nn
import torch
from evaluators.evaluator import Evaluator
from trainers.tf_trainer import TransformerTrainer, MyLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime
from tables.generate_n_token_fig import generate_n_token_fig


def transformer_main(model_name, model_class, model_dataset, args):
    # Check if we're loading and evaluating a model, or training and evaluating
    eval_only = args.eval_only
    if eval_only:
        print(f"Loading and evaluating a {model_name} model")
    else:
        print(f"Training and evaluating a {model_name} model")

    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    # Set Device
    if args.cuda:
        args.device = torch.device('cuda:0')
        print("Using a CUDA GPU, woot!")
    else:
        args.device = 'cpu'
        print("Using a CPU, sad!")

    config = deepcopy(args)
    config.run_name = model_name + "_" + start_time

    # Set CUDA Blocking if needed, used when getting CUDA errors:
    if config.cuda_block:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    else:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

    # Loss and optimizer
    if config.imbalance_fix == 'loss_weight':
        dataset = model_dataset(config)
        target_perc = dataset.get_class_balance()  # Percentage of targets = 1
        pos_weight = (1 - target_perc) / target_perc
        print(f"Weighting our Loss Function to Balance Target Classes\n"
              f"Training examples with target=1 will get a factor of: {round(pos_weight, 3)}")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    elif config.imbalance_fix == 'none':
        loss_fn = nn.BCEWithLogitsLoss()
    elif config.imbalance_fix == 'undersampling':
        loss_fn = nn.BCEWithLogitsLoss()
        dataset = model_dataset(config, undersample=True)
    else:
        raise Exception("Invalid method to fix the class imbalance provided, or not yet implemented")

    # If we're just running this to count tokens in our documents, exit this script and call relevant script
    if config.count_tokens:
        generate_n_token_fig(model_name, config, dataset)
        sys.exit()

    # Instantiate our Model
    steps_per_epoch = dataset.get_n_training() / config.batch_size
    model = model_class(config, loss_fn, steps_per_epoch)

    # Make and create if needed dir for loggers to log to
    config.results_dir_target = os.path.join(config.results_dir, config.target)  # dir for a targets results
    config.results_dir_model = os.path.join(config.results_dir_target, model_name)  # subdir for each model

    if not os.path.exists(config.results_dir_target):
        os.mkdir(config.results_dir_target)
    if not os.path.exists(config.results_dir_model):
        os.mkdir(config.results_dir_model)

    # Try out some different loggers
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=config.results_dir_model)
    my_logger = MyLogger(save_dir=config.results_dir_model)

    # Save Checkpoint
    # saves a file like: input/BERT-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_bal',  # monitored quantity
        filename=f'{tb_logger.log_dir}/' + model_name + '--{epoch}_val_bal_{val_bal:.2f}',
        save_top_k=1,  # save the top 1 model1
        mode='max',  # mode of the monitored quantity  for optimization
        dirpath=config.results_dir_model
    )

    trainer = TransformerTrainer(config=config,
                                 checkpoint_callback=checkpoint_callback,
                                 logger=[tb_logger, my_logger]
                                 )
    # Load and evaluate vs train and evaluate
    if eval_only:
        loaded_model = model_class.load_from_checkpoint(
            checkpoint_path=config.model_file,
            hparams_file=r"C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\BERT\hparams.yaml",
            map_location=None)
        trainer.test(loaded_model, dataloaders=dataset)
    else:
        trainer.fit(model, dataset)
        trainer.test(model, dataset)
    train_history, dev_history, test_history, start_time = trainer.get_results()
    evaluator = Evaluator(model_name, test_history, config, start_time)

    # Write the run history, and update the master results file
    evaluator.write_result_history()
    evaluator.append_to_results()

    quit("All done!")
