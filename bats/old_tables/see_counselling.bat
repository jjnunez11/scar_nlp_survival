:: tbl_see_counselling
:: Performance predicting whether a patient will see a counsellor (SOCIALWORK) in first 5 years (60 months)
python -m models.bow --target "dsplnic_SOCIALWORK_60" --table "see_counselling" --epochs 1
python -m models.cnn --target "dsplnic_SOCIALWORK_60" --table "see_counselling"
python -m models.lstm --target "dsplnic_SOCIALWORK_60" --table "see_counselling"
python -m models.bert --target "dsplnic_SOCIALWORK_60" --table "see_counselling"

:: RAN