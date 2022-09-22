:: python -m models.bert --target "surv_mo_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0
:: python -m models.bert --target "surv_mo_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0.01
::python -m models.bert --target "surv_mo_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0.1

python -m models.bert --target "surv_mo_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0.99
python -m models.bert --target "surv_mo_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0.9
python -m models.bert --target "surv_mo_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0.7
python -m models.bert --target "surv_mo_60" --table "bert_tuning"  --lr 0.00005 --weight-decay 0.4







:: then do surv_mo_36 and surv_mo_6