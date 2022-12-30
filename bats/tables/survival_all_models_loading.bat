:: tbl_survival_all_models
:: Performance predicting whether a patient will survive 60 months after
:: the document models are trained on was generated
:: this loads previously trained models
python -m models.bow --target "surv_mo_60" --table "survival_all_models" --epochs 1 --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\BoW\BoW_20220928-2054_e0.pbz2
python -m models.cnn --target "surv_mo_60" --table "survival_all_models" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\CNN\CNN_20220928-2210.pt
python -m models.lstm --target "surv_mo_60" --table "survival_all_models" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\LSTM\LSTM_20220929-0009.pt
python -m models.bert --target "surv_mo_60" --batch-size 8 --table "survival_all_models" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\BERT\BERT--epoch=11_val_bal_val_bal=0.80.ckpt --hparams-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\BERT\hparams.yaml
