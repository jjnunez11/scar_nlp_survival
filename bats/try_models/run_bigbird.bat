:: Working
:: full epoch tested
:: python -m models.bigbird --target "need_emots_1" --block-size 16 --num-blocks 3 --max_tokens 64
:: trained awhile
:: python -m models.bigbird --target "need_emots_1" --block-size 16 --num-blocks 3 --max_tokens 128


:: Trial
:: python -m models.bigbird --target "need_emots_1" --batch-size 1 OUT OF MEMORY AT EPOCH 2
:: python -m models.bigbird --target "need_emots_1" --batch-size 1 --cuda-block True 52h per epoch, can't increase batch-size
:: python -m models.bigbird --target "need_emots_1" --batch-size 1 --gradient-checkpointing True | Failed in 3rd epoch

:: python -m models.bigbird --target "dspln_PSYCHIATRY_0"
:: python -m models.bigbird --target "dspln_SOCIALWORK_0"
:: python -m models.bigbird --target "need_emots_1"
:: python -m models.bigbird --target "need_emots_2"
:: python -m models.bigbird --target "need_infos_1"
:: python -m models.bigbird --target "need_infos_2"
:: python -m models.bigbird --target "surv_mo_6"
:: python -m models.bigbird --target "surv_mo_36"
:: python -m models.bigbird --target "surv_mo_60"

:: python -m models.bigbird --target "dsplnmulti_SOCIALWORK_2"
:: python -m models.bigbird --target "dsplnmulti_PSYCHIATRY_2"