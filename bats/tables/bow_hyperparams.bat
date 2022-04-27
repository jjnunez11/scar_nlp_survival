:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 0.1
:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 1
:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 2

:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 0.1 --max-tokens 5000
:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 0.2 --max-tokens 5000
:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 0.3 --max-tokens 5000
:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 0.5 --max-tokens 5000
:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 1 --max-tokens 5000

python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 0.1 --max-tokens 10000
python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 0.5 --max-tokens 10000
python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "bow_hyperparams" --epochs 1 --classifier "l2logreg" --l2logreg-c 0.01 --max-tokens 10000

:: RAN AND IN TABLE
