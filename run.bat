::python -m models.longformer --target "need_emots_1" --load-ckpt "C:\Users\jjnunez\PycharmProjects\scar_nlp\results\need_emots_1\Longformer\default\version_38\Longformer--epoch=22_val_bal_val_bal=0.62.ckpt"
:: python -m models.longformer --target "surv_mo_60" --max-tokens 2048 --batch-size 1

:: python -m models.longformer --target "need_emots_1" --max-tokens 2048

:: python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_0" --epochs 100
:: python -m models.bert --lr 0.0001 --target "dspln_PSYCHIATRY_0" --epochs 1 --batch-size 8 --max-tokens 50

:: python -m models.bert --lr 0.0001 --target "dspln_PSYCHIATRY_0" --epochs 1 --batch-size 8 --max-tokens 50 --imbalance-fix "undersampling"
:: python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_0" --epochs 1 --imbalance-fix "undersampling"

:: python -m models.longformer --target "dspln_PSYCHIATRY_0" --epochs 1 --imbalance-fix "undersampling" --batch-size 1

:: python -m models.lstm --lr 0.00001 --target "dspln_PSYCHIATRY_0" --epochs 1 --imbalance-fix "undersampling"

:: python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_0" --patience 10
:: python -m models.cnn --lr 0.0001 --target "dspln_PSYCHIATRY_0"
:: python -m models.cnn --lr 0.000001 --target "dspln_PSYCHIATRY_0"
:: python -m models.cnn --lr 0.0000001 --target "dspln_PSYCHIATRY_0"

:: python -m models.lstm --lr 0.00001 --target "dspln_PSYCHIATRY_0" --epochs 1
:: python -m models.cnn --lr 0.00001 --target "surv_mo_60" --patience 10 --epochs 100

:: python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_0" --table "see_psych" --epochs 1

:: python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_0" --table "see_psych"
:: python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_12" --table "see_psych"


:: python -m models.cnn --lr 0.000001 --target "dspln_PSYCHIATRY_60" --table "see_psych" --patience 10 --epochs 1000
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "rf"
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "elnet"
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "gbdt"
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 5
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 5

:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 500
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 1000
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 2000
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 5000
:: python -m models.bow --target "dspln_PSYCHIATRY_60" --table "see_psych" --epochs 1 --classifier "l2logreg" --max-tokens 10000

:: python -m models.bow --target "dspln_PSYCHIATRY_60" --epochs 1 --classifier "l2logreg" --max-tokens 100 --l2logreg-c 1 --table "test"
:: python -m models.cnn --target "dsplnic_PSYCHIATRY_60" --table "test" --batch-size 16
:: python -m models.lstm --target "dsplnic_PSYCHIATRY_60" --table "test" --batch-size 64
:: python -m models.bert --target "dsplnic_PSYCHIATRY_60" --table "test" --epochs 2

:: python -m models.bow --target "survic_mo_60" --epochs 1
:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --epochs 1
:: python -m models.bow --target "dsplnic_SOCIALWORK_60" --epochs 1
:: python -m models.bow --target "need_emots_4" --epochs 1

:: python -m models.longformer --target "dsplnic_PSYCHIATRY_60" --max-tokens 128 --imbalance-fix "undersampling"
:: python -m models.lstm --target "dsplnic_PSYCHIATRY_60" --table "test" --batch-size 64 --epochs 1
:: python -m models.lstm --target "dsplnic_PSYCHIATRY_60" --table "test" --batch-size 32 --epochs 1

:: Re-run LSTM models after fixing to match prior dsplnic_SOCIALWORK_60
:: python -m models.lstm --target "need_emots_4" --table "need_emots_all_models"
:: python -m models.lstm --target "need_infos_4" --table "need_infos_all_models"
:: python -m models.lstm --target "dsplnic_PSYCHIATRY_60" --table "see_psych"
:: python -m models.lstm --target "dsplnic_SOCIALWORK_60" --table "see_counselling"
:: python -m models.lstm --target "survic_mo_60" --table "survival_all_models"

:: python -m models.cnn --target "dsplnic_PSYCHIATRY_60"

:: python -m models.bow --target "surv_mo_60" --epochs 1
:: python -m models.cnn --target "surv_mo_60" --epochs 1
