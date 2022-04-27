:: python -m models.bert --target "survic_mo_60" --table "compare_token_len_survic" --max-tokens 128 --table_extra 128 --imbalance-fix "undersampling"
:: python -m models.bert --target "survic_mo_60" --table "compare_token_len_survic" --max-tokens 256 --table_extra 256 --imbalance-fix "undersampling"
:: python -m models.bert --target "survic_mo_60" --table "compare_token_len_survic" --max-tokens 512 --table_extra 512 --imbalance-fix "undersampling"

:: python -m models.longformer --target "survic_mo_60" --table "compare_token_len_survic" --max-tokens 256 --table_extra 256 --imbalance-fix "undersampling" --lr 0.001
:: python -m models.longformer --target "survic_mo_60" --table "compare_token_len_survic" --max-tokens 512 --table_extra 512 --imbalance-fix "undersampling" --lr 0.001
python -m models.longformer --target "survic_mo_60" --table "compare_token_len_survic" --max-tokens 1024 --table_extra 1024 --imbalance-fix "undersampling"  --lr 0.001
python -m models.longformer --target "survic_mo_60" --table "compare_token_len_survic" --max-tokens 2048 --table_extra 2048 --imbalance-fix "undersampling" --lr 0.001
python -m models.longformer --target "survic_mo_60" --table "compare_token_len_survic" --max-tokens 4096 --table_extra 4096 --imbalance-fix "undersampling" --lr 0.001