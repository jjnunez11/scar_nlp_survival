:: tbl_need_infos_all_models
:: predicting if patients have a certain number of info needs
python -m models.bow --target "need_infos_4" --table "need_infos_all_models" --epochs 1
python -m models.cnn --target "need_infos_4" --table "need_infos_all_models"
python -m models.lstm --target "need_infos_4" --table "need_infos_all_models"
python -m models.bert --target "need_infos_4" --batch-size 8 --table "need_infos_all_models"

:: RUNNING