python -m models.lstm --dataset SCAR --mode rand --batch-size 32 --lr 0.0001 --epochs 100 ^
 --num-layers 2 --hidden-dim 256 --data-version "PPv3" --target "emots"

:: Still need to run the following, but should work. May be able to use higher batch_size

python -m models.lstm --dataset SCAR --mode rand --batch-size 32 --lr 0.0001 --epochs 100 ^
 --num-layers 2 --hidden-dim 256 --data-version "PPv3" --target "infos"

python -m models.lstm --dataset SCAR --mode rand --batch-size 32 --lr 0.0001 --epochs 100 ^
 --num-layers 2 --hidden-dim 256 --data-version "PPv3" --target "see_SOCIALWORK_by_999mo"

python -m models.lstm --dataset SCAR --mode rand --batch-size 32 --lr 0.0001 --epochs 100 ^
 --num-layers 2 --hidden-dim 256 --data-version "PPv3" --target "36mo_err_surv"

