:: Working
:: python -m models.longformer --target "need_emots_1" --attention-window=64 --max-tokens 128
:: python -m models.longformer --target "need_emots_1" --attention-window=512 --max-tokens 512
:: python -m models.longformer --target "need_emots_1" --attention-window=512 --max-tokens
:: python -m models.longformer --target "need_emots_1"  --attention-window=512 --max-tokens 514 --batch-size 2
:: python -m models.longformer --target "need_emots_1"  --attention-window=512 --max-tokens 1024 --batch-size 2
:: python -m models.longformer --target "need_emots_1"  --attention-window=512 --max-tokens 2048 --batch-size 1
:: python -m models.longformer --target "need_emots_1"  --attention-window=512 --max-tokens 4096 --batch-size 1


:: Trial
python -m models.longformer --target "need_emots_1"  --batch-size 1 --max-tokens 2048
:: python -m models.longformer --target "dspln_PSYCHIATRY_0"
:: python -m models.longformer --target "dspln_SOCIALWORK_0"
:: python -m models.longformer --target "need_emots_1"
:: python -m models.longformer --target "need_emots_2"
:: python -m models.longformer --target "need_infos_1"
:: python -m models.longformer --target "need_infos_2"
:: python -m models.longformer --target "surv_mo_6"
:: python -m models.longformer --target "surv_mo_36"
:: python -m models.longformer --target "surv_mo_60"

:: python -m models.longformer --target "dsplnmulti_SOCIALWORK_2"
:: python -m models.longformer --target "dsplnmulti_PSYCHIATRY_2"