:: tbl_need_emots_dif_n
:: Performance predicting how long a patient will survive at different numbers of months
python -m models.bow --target "need_emots_1" --table "need_emots_dif_n" --epochs 1 --table_extra "1"
python -m models.bow --target "need_emots_2" --table "need_emots_dif_n" --epochs 1 --table_extra "2"
python -m models.bow --target "need_emots_3" --table "need_emots_dif_n" --epochs 1 --table_extra "3"
python -m models.bow --target "need_emots_4" --table "need_emots_dif_n" --epochs 1 --table_extra "4"
python -m models.bow --target "need_emots_5" --table "need_emots_dif_n" --epochs 1 --table_extra "5"

python -m models.cnn --target "need_emots_1" --table "need_emots_dif_n" --table_extra "1"
python -m models.cnn --target "need_emots_2" --table "need_emots_dif_n" --table_extra "2"
python -m models.cnn --target "need_emots_3" --table "need_emots_dif_n" --table_extra "3"
python -m models.cnn --target "need_emots_4" --table "need_emots_dif_n" --table_extra "4"
python -m models.cnn --target "need_emots_5" --table "need_emots_dif_n" --table_extra "5"

:: RAN SUCCESFULLY