:: tbl_survival_dif_lengths_multi
:: Performance predicting how long a patient will survive at different numbers of months, repeating for variance
:: python -m models.bow --target "surv_mo_6" --table "survival_dif_lengths_multi" --epochs 1 --table_extra "6"
:: python -m models.bow --target "surv_mo_36" --table "survival_dif_lengths_multi" --epochs 1 --table_extra "36"
:: python -m models.bow --target "surv_mo_60" --table "survival_dif_lengths_multi" --epochs 1 --table_extra "60"

:: python -m models.cnn --target "surv_mo_6" --table "survival_dif_lengths_multi" --table_extra "6"
:: python -m models.cnn --target "surv_mo_36" --table "survival_dif_lengths_multi" --table_extra "36"
:: python -m models.cnn --target "surv_mo_60" --table "survival_dif_lengths_multi" --table_extra "60"

:: python -m models.lstm --target "surv_mo_6" --table "survival_dif_lengths_multi" --table_extra "6"
:: python -m models.lstm --target "surv_mo_36" --table "survival_dif_lengths_multi" --table_extra "36"
python -m models.lstm --target "surv_mo_60" --table "survival_dif_lengths_multi" --table_extra "60"

python -m models.bert --target "surv_mo_6" --batch-size 8 --table "survival_dif_lengths_multi_extra" --table_extra "6"
python -m models.bert --target "surv_mo_36" --batch-size 8 --table "survival_dif_lengths_multi_extra" --table_extra "36"
python -m models.bert --target "surv_mo_60" --batch-size 8 --table "survival_dif_lengths_multi_extra" --table_extra "60"