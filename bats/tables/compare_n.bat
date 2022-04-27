:: compare see_psych prediction with different numbers of subjects
python -m models.bow --target "dsplnic_PSYCHIATRY_60_3000" --table "compare_n" --epochs 1 --table_extra "3000"
python -m models.cnn --target "dsplnic_PSYCHIATRY_60_3000" --table "compare_n" --table_extra "3000"
python -m models.bow --target "dsplnic_PSYCHIATRY_60_4000" --table "compare_n" --epochs 1 --table_extra "4000"
python -m models.cnn --target "dsplnic_PSYCHIATRY_60_4000" --table "compare_n" --table_extra "4000"
python -m models.bow --target "dsplnic_PSYCHIATRY_60_6000" --table "compare_n" --epochs 1 --table_extra "6000"
python -m models.cnn --target "dsplnic_PSYCHIATRY_60_6000" --table "compare_n" --table_extra "6000"
python -m models.bow --target "dsplnic_PSYCHIATRY_60_9000" --table "compare_n" --epochs 1 --table_extra "9000"
python -m models.cnn --target "dsplnic_PSYCHIATRY_60_9000" --table "compare_n" --table_extra "9000"
python -m models.bow --target "dsplnic_PSYCHIATRY_60_12000" --table "compare_n" --epochs 1 --table_extra "12000"
python -m models.cnn --target "dsplnic_PSYCHIATRY_60_12000" --table "compare_n" --table_extra "12000"
python -m models.bow --target "dsplnic_PSYCHIATRY_60_15000" --table "compare_n" --epochs 1 --table_extra "15000"
python -m models.cnn --target "dsplnic_PSYCHIATRY_60_15000" --table "compare_n" --table_extra "15000"
python -m models.bow --target "dsplnic_PSYCHIATRY_60_20000" --table "compare_n" --epochs 1 --table_extra "20000"
python -m models.cnn --target "dsplnic_PSYCHIATRY_60_20000" --table "compare_n" --table_extra "20000"
python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "compare_n" --epochs 1 --table_extra "30953"
python -m models.cnn --target "dsplnic_PSYCHIATRY_60" --table "compare_n" --table_extra "30953"

:: TO RUN