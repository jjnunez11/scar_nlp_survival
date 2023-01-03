:: tbl_survival_dif_lengths
:: Performance predicting how long a patient will survive at different numbers of months
:: Evaluates previously trained models
python -m models.bow --target "surv_mo_6" --table "survival_dif_lengths" --epochs 1 --table_extra "6" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_6\BoW\BoW_20220928-2004_e0.pbz2
python -m models.bow --target "surv_mo_36" --table "survival_dif_lengths" --epochs 1 --table_extra "36" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_36\BoW\BoW_20220928-2029_e0.pbz2
python -m models.bow --target "surv_mo_60" --table "survival_dif_lengths" --epochs 1 --table_extra "60" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\BoW\BoW_20220928-2334_e0.pbz2

python -m models.cnn --target "surv_mo_6" --table "survival_dif_lengths" --table_extra "6" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_6\CNN\CNN_20220928-2119.pt
python -m models.cnn --target "surv_mo_36" --table "survival_dif_lengths" --table_extra "36" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_36\CNN\CNN_20220928-2141.pt
python -m models.cnn --target "surv_mo_60" --table "survival_dif_lengths" --table_extra "60" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\CNN\CNN_20220928-2342.pt

python -m models.lstm --target "surv_mo_6" --table "survival_dif_lengths" --table_extra "6" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_6\LSTM\LSTM_20220929-1501.pt
python -m models.lstm --target "surv_mo_36" --table "survival_dif_lengths" --table_extra "36" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_36\LSTM\LSTM_20220929-1840.pt
python -m models.lstm --target "surv_mo_60" --table "survival_dif_lengths" --table_extra "60" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\LSTM\LSTM_20220929-2219.pt

python -m models.bert --target "surv_mo_6" --batch-size 8 --table "survival_dif_lengths" --table_extra "6" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_6\BERT\version_2\BERT--epoch=13_val_bal_val_bal=0.84.ckpt --hparams-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_6\BERT\version_2\hparams.yaml
python -m models.bert --target "surv_mo_36" --batch-size 8 --table "survival_dif_lengths" --table_extra "36" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_36\BERT\version_2\BERT--epoch=15_val_bal_val_bal=0.83.ckpt --hparams-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_36\BERT\version_2\hparams.yaml
python -m models.bert --target "surv_mo_60" --batch-size 8 --table "survival_dif_lengths" --table_extra "60" --eval_only --model-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\BERT\BERT--epoch=19_val_bal_val_bal=0.81.ckpt --hparams-file C:\Users\jjnunez\PycharmProjects\scar_nlp_survival\results\paper_submission\surv_mo_60\BERT\hparams_20221001.yaml
