:: All need hyperparameter tuning; elnet doesn't actually train yet

:: python -m models.bow --dataset SCAR --epochs 10 --classifier "rf" --imbalance-fix "loss_weight"
:: python -m models.bow --dataset SCAR --epochs 10 --classifier "elnet" --imbalance-fix "loss_weight"
:: python -m models.bow --dataset SCAR --epochs 10 --classifier "gbdt" --imbalance-fix "none"
:: python -m models.bow --dataset SCAR --epochs 10 --classifier "l2logreg" --imbalance-fix "loss_weight"

:: Try different targets
python -m models.bow --epochs 10 --target "dspln_PSYCHIATRY_0"
python -m models.bow --epochs 10 --target "dspln_SOCIALWORK_0"
python -m models.bow --epochs 10 --target "need_emots_1"
python -m models.bow --epochs 10 --target "need_emots_2"
python -m models.bow --epochs 10 --target "need_infos_1"
python -m models.bow --epochs 10 --target "need_infos_2"
python -m models.bow --epochs 10 --target "surv_mo_6"
python -m models.bow --epochs 10 --target "surv_mo_36"
python -m models.bow --epochs 10 --target "surv_mo_60"

python -m models.bow --epochs 10 --target "dsplnmulti_SOCIALWORK_2"
python -m models.bow --epochs 10 --target "dsplnmulti_PSYCHIATRY_2"