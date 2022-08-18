:: tbl_survival_dif_lengths
:: Performance predicting how long a patient will survive at different numbers of months
:: python -m models.bow --target "surv_mo_6" --table "survival_dif_lengths" --epochs 1 --table_extra "6_1095afterdx"
:: python -m models.bow --target "surv_mo_36" --table "survival_dif_lengths" --epochs 1 --table_extra "36_1095afterdx"
:: python -m models.bow --target "surv_mo_60" --table "survival_dif_lengths" --epochs 1 --table_extra "60_1095afterdx"

:: python -m models.cnn --target "surv_mo_6" --table "survival_dif_lengths" --table_extra "6_1095afterdx"
:: python -m models.cnn --target "surv_mo_36" --table "survival_dif_lengths" --table_extra "36_1095afterdx"
:: python -m models.cnn --target "surv_mo_60" --table "survival_dif_lengths" --table_extra "60_1095afterdx"

:: RUNNING

python -m models.cnn --target "surv_mo_60" --table "survival_dif_lengths" --table_extra "60_180afterdx"
python -m models.cnn --target "surv_mo_60" --table "survival_dif_lengths" --table_extra "60_180afterdx"
python -m models.cnn --target "surv_mo_60" --table "survival_dif_lengths" --table_extra "60_180afterdx"
python -m models.cnn --target "surv_mo_60" --table "survival_dif_lengths" --table_extra "60_180afterdx"
