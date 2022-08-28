:: Compare the tuned LSTM vs the base LSTM on all three survival lengths

:: python -m models.lstm --target "surv_mo_60" --table "lstm_tuning"
python -m models.lstm --target "surv_mo_60" --table "lstm_tuning"  --lr 0.00005 --dropout 0.2 --wdrop 0.3 --embed-droprate 0.2

:: then do surv_mo_36 and surv_mo_6