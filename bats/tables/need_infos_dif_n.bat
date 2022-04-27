:: tbl_need_infos_dif_n
:: Performance predicting if patients have various numbers of information needs
python -m models.bow --target "need_infos_1" --table "need_infos_dif_n" --epochs 1 --table_extra "1"
python -m models.bow --target "need_infos_2" --table "need_infos_dif_n" --epochs 1 --table_extra "2"
python -m models.bow --target "need_infos_3" --table "need_infos_dif_n" --epochs 1 --table_extra "3"
python -m models.bow --target "need_infos_4" --table "need_infos_dif_n" --epochs 1 --table_extra "4"

python -m models.cnn --target "need_infos_1" --table "need_infos_dif_n" --table_extra "1"
python -m models.cnn --target "need_infos_2" --table "need_infos_dif_n" --table_extra "2"
python -m models.cnn --target "need_infos_3" --table "need_infos_dif_n" --table_extra "3"
python -m models.cnn --target "need_infos_4" --table "need_infos_dif_n" --table_extra "4"

:: READY TO RUN