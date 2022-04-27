:: tbl_survival_all_models
:: Performance predicting whether a patient will survive 60 months after
:: the document models are trained on was generated
python -m models.bow --target "survic_mo_60" --table "survival_all_models" --epochs 1
python -m models.cnn --target "survic_mo_60" --table "survival_all_models"
python -m models.lstm --target "survic_mo_60" --table "survival_all_models"
python -m models.bert --target "survic_mo_60" --batch-size 8 --table "survival_all_models"

:: READY TO RUN