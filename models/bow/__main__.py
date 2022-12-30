import os
from copy import deepcopy
from trainers.bow_trainer import BoWTrainer
from evaluators.evaluator import Evaluator
from models.bow.args import get_args
from datasets.scar_bow import SCARBoW
import warnings
import datetime

if __name__ == '__main__':

    args = get_args()

    # Setup config
    config = deepcopy(args)
    model_name = "BoW"
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    config.run_name = model_name + "_" + start_time

    eval_only = args.eval_only
    if eval_only:
        print(f"Loading and evaluating a {model_name} model")
    else:
        print(f"Training and evaluating a {model_name} model")

    # Loss
    if config.imbalance_fix == 'loss_weight':
        class_weight = 'balanced'

        if config.classifier == "gbdt":
            warnings.warn("sklearn does not have class_weight, so can't use loss_weight to balance"
                          "gbdt, setting to none", stacklevel=2)
            # Set config's imbalance_fix to none for gbdt, so when results are written out, we see
            # that loss weighting wasn't used.
            config_dict = vars(config)
            config_dict['imbalance_fix'] = 'none'

        scar_bow = SCARBoW(args)  # args.batch_size, args.data_dir, args.target)
    elif args.imbalance_fix == 'undersampling':
        class_weight = None
        scar_bow = SCARBoW(args, undersample=True)
    elif config.imbalance_fix == 'none':
        class_weight = None
    else:
        raise Exception("Invalid method to fix the class imbalance provided, or not yet implemented")

    # Load the data
    train_data = scar_bow.get_train_data()
    dev_data = scar_bow.get_dev_data()
    test_data = scar_bow.get_test_data()

    # Make directories for results if not already there
    config.results_dir_target = os.path.join(config.results_dir, config.target)  # dir for a targets results
    config.results_dir_model = os.path.join(config.results_dir_target, model_name)  # subdir for each model

    if not os.path.exists(config.results_dir_target):
        os.mkdir(config.results_dir_target)
    if not os.path.exists(config.results_dir_model):
        os.mkdir(config.results_dir_model)

    # Train and Evaluate Model
    trainer = BoWTrainer(config=config, class_weight=class_weight)
    if eval_only:
        test_history = trainer.eval_only(test_data)
    else:
        train_history, dev_history, test_history, start_time = trainer.fit(train_data, dev_data, test_data, config.epochs)
    evaluator = Evaluator("BoW", test_history, config, start_time)

    # Use evaluator to print the best epochs
    print('\nBest epoch for AUC:')
    evaluator.print_best_auc()

    print('\nBest epoch for F1:')
    evaluator.print_best_f1()

    # Write the run history, and update the master results file
    evaluator.write_result_history()
    evaluator.append_to_results()
