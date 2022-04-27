:: Stage I vs IV

:: python -m models.bow --epochs 10 --target "need_emots_1_I"
:: python -m models.bow --epochs 10 --target "need_emots_1_IV"

:: python -m models.lstm --batch-size 32 --target "need_emots_1_I"
:: python -m models.lstm --batch-size 32 --target "need_emots_1_IV"

:: python -m models.bert --target "need_emots_1_I"
:: python -m models.bert --target "need_emots_1_IV"

:: python -m models.cnn --lr 0.00001 --target "need_emots_1_I"
:: python -m models.cnn --lr 0.00001 --target "need_emots_1_IV"

:: python -m models.bow --epochs 10 --target "surv_mo_60_I"
:: python -m models.bow --epochs 10 --target "surv_mo_60_IV"

:: python -m models.lstm --batch-size 32 --target "surv_mo_60_I"
:: python -m models.lstm --batch-size 32 --target "surv_mo_60_IV"

:: python -m models.bert --target "surv_mo_60_I"
:: python -m models.bert --target "surv_mo_60_IV"

:: python -m models.cnn --lr 0.00001 --target "surv_mo_60_I"
:: python -m models.cnn --lr 0.00001 --target "surv_mo_60_IV"

:: Metastatic vs Non-metastatic

python -m models.bow --epochs 10 --target "need_emots_1_nonmet"
python -m models.bow --epochs 10 --target "need_emots_1_met"

python -m models.lstm --batch-size 32 --target "need_emots_1_nonmet"
python -m models.lstm --batch-size 32 --target "need_emots_1_met"

python -m models.bert --target "need_emots_1_nonmet"
python -m models.bert --target "need_emots_1_met"

python -m models.cnn --lr 0.00001 --target "need_emots_1_nonmet"
python -m models.cnn --lr 0.00001 --target "need_emots_1_met"

python -m models.bow --epochs 10 --target "surv_mo_60_nonmet"
python -m models.bow --epochs 10 --target "surv_mo_60_met"

python -m models.lstm --batch-size 32 --target "surv_mo_60_nonmet"
python -m models.lstm --batch-size 32 --target "surv_mo_60_met"

python -m models.bert --target "surv_mo_60_nonmet"
python -m models.bert --target "surv_mo_60_met"

python -m models.cnn --lr 0.00001 --target "surv_mo_60_nonmet"
python -m models.cnn --lr 0.00001 --target "surv_mo_60_met"