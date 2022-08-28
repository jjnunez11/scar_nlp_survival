:: tbl_need_emots_all_models
:: Performance predicting whether a patient has given emotional needs
python -m models.bow --target "need_emots_4" --table "need_emots_all_models" --epochs 1
python -m models.cnn --target "need_emots_4" --table "need_emots_all_models"
python -m models.lstm --target "need_emots_4" --table "need_emots_all_models"
python -m models.bert --target "need_emots_4" --batch-size 8 --table "need_emots_all_models"

:: RAN