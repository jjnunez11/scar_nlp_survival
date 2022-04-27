:: Working Runs of Kim CNN on different targets

::python -m models.cnn --mode rand --dataset SCAR --batch-size 8 --lr 0.00001 --epochs 100 ^
:: --embed-dim 300 --words-dim 300 --output-channel 500 --data-version "PPv3" --target "emots"

:: python -m models.cnn --mode rand --dataset SCAR --batch-size 8 --lr 0.00001 --epochs 100 ^
:: --embed-dim 300 --words-dim 300 --output-channel 500 --data-version "PPv3" --target "infos"

:: python -m models.cnn --mode rand --dataset SCAR --batch-size 8 --lr 0.00001 --epochs 100 ^
:: --embed-dim 300 --words-dim 300 --output-channel 500 --data-version "PPv3" --target "see_SOCIALWORK_by_999mo"

:: python -m models.cnn --mode rand --dataset SCAR --batch-size 8 --lr 0.00001 --epochs 100 ^
:: --embed-dim 300 --words-dim 300 --output-channel 500 --data-version "PPv3" --target "36mo_err_surv"

:: Try different targets

python -m models.cnn --lr 0.00001 --target "dspln_PSYCHIATRY_0"
python -m models.cnn --lr 0.00001 --target "dspln_SOCIALWORK_0"
python -m models.cnn --lr 0.00001 --target "need_emots_1"
python -m models.cnn --lr 0.00001 --target "need_emots_2"
python -m models.cnn --lr 0.00001 --target "need_infos_1"
python -m models.cnn --lr 0.00001 --target "need_infos_2"
python -m models.cnn --lr 0.00001 --target "surv_mo_6"
python -m models.cnn --lr 0.00001 --target "surv_mo_36"
python -m models.cnn --lr 0.00001 --target "surv_mo_60"

python -m models.cnn --lr 0.00001 --target "dsplnmulti_SOCIALWORK_2"
python -m models.cnn --lr 0.00001 --target "dsplnmulti_PSYCHIATRY_2"