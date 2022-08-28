python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "imbalance-fix" --epochs 1 --table "imbalance_fix" --table_extra "Undersampling"
:: python -m models.cnn --lr 0.0001 --target "dsplnic_PSYCHIATRY_60" --imbalance-fix "undersampling" --table "imbalance_fix" --table_extra "Undersampling"
:: python -m models.lstm --lr 0.0001 --target "dsplnic_PSYCHIATRY_60"  --imbalance-fix "undersampling" --table "imbalance_fix" --table_extra "Undersampling"
:: python -m models.bert --lr 0.0001 --target "dsplnic_PSYCHIATRY_60"  --imbalance-fix "undersampling" --table "imbalance_fix" --table_extra "Undersampling"

:: python -m models.bow --target "dsplnic_PSYCHIATRY_60" --table "imbalance_fix" --epochs 1 --table_extra "Weighted Loss"
:: python -m models.cnn --lr 0.0001 --target "dsplnic_PSYCHIATRY_60" --table "imbalance_fix" --table_extra "Weighted Loss"
:: python -m models.lstm --lr 0.0001 --target "dsplnic_PSYCHIATRY_60" --table "imbalance_fix" --table_extra "Weighted Loss"
:: python -m models.bert --lr 0.0001 --target "dsplnic_PSYCHIATRY_60"  --table "imbalance_fix" --table_extra "Weighted Loss"

