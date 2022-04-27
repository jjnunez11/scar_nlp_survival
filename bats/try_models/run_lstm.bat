:: python -m models.lstm -batch-size 32 --dataset SCAR --mode rand --batch-size 32 --lr 0.0001 --epochs 100 ^
::  --num-layers 2 --hidden-dim 128 --data-version "PPv3" --target "emots"

:: Still need to run the following, but should work. May be able to use higher batch_size

:: python -m models.lstm -batch-size 32 --dataset SCAR --mode rand --batch-size 32 --lr 0.0001 --epochs 100 ^
:: --num-layers 2 --hidden-dim 128 --data-version "PPv3" --target "infos"

:: python -m models.lstm -batch-size 32 --dataset SCAR --mode rand --batch-size 32 --lr 0.0001 --epochs 100 ^
:: --num-layers 2 --hidden-dim 128 --data-version "PPv3" --target "see_SOCIALWORK_by_999mo"

:: python -m models.lstm -batch-size 32 --dataset SCAR --mode rand --batch-size 32 --lr 0.0001 --epochs 100 ^
::  --num-layers 2 --hidden-dim 128 --data-version "PPv3" --target "36mo_err_surv"

:: Try different targets
python -m models.lstm --batch-size 32 --target "dspln_PSYCHIATRY_0"
python -m models.lstm --batch-size 32 --target "dspln_SOCIALWORK_0"
python -m models.lstm --batch-size 32 --target "need_emots_1"
python -m models.lstm --batch-size 32 --target "need_emots_2"
python -m models.lstm --batch-size 32 --target "need_infos_1"
python -m models.lstm --batch-size 32 --target "need_infos_2"
python -m models.lstm --batch-size 32 --target "surv_mo_6"
python -m models.lstm --batch-size 32 --target "surv_mo_36"
python -m models.lstm --batch-size 32 --target "surv_mo_60"

python -m models.lstm --batch-size 32 --target "dsplnmulti_SOCIALWORK_2"
python -m models.lstm --batch-size 32 --target "dsplnmulti_PSYCHIATRY_2"