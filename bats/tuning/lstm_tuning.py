from subprocess import check_output

# Hyper-parameters
table = "lstm_tuning"
target = "surv_mo_60"
# Do we need bidirectional? Is it working wit bool?
num_layers = [1]
hidden_dim = [256, 512]
dropouts = [0.1, 0.2, 0.5, 0.8]
embed_droprates = [0.05, 0.1, 0.15]
wdrops = [0.1, 0.2, 0.3]
lrs = [0.00005]

for num_layer in num_layers:
    for dropout in dropouts:
        for embed_droprate in embed_droprates:
            for wdrop in wdrops:
                for lr in lrs:
                    command = f'python -m models.lstm --target {target} ' \
                              f'--table {table} --num-layers {num_layer}  --lr {lr}' \
                              f' --dropout {dropout} --embed-droprate {embed_droprate} --wdrop {wdrop}'
                    print(f'executing command: {command}')
                    check_output(command, shell=True)
