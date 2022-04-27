python -m models.bow --target "dsplnic_PSYCHIATRY_60_male" --table "compare_sexes" --epochs 1 --table_extra "Male & 14711"
python -m models.cnn --target "dsplnic_PSYCHIATRY_60_male" --table "compare_sexes" --table_extra "Male & 14711"
python -m models.bow --target "dsplnic_PSYCHIATRY_60_female" --table "compare_sexes" --epochs 1 --table_extra "Female & 16242"
python -m models.cnn --target "dsplnic_PSYCHIATRY_60_female" --table "compare_sexes" --table_extra "Female & 16242"


:: DONE