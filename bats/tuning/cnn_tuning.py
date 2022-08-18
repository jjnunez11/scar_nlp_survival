from subprocess import check_output

# Hyper-parameters
table = "cnn_tuning"
target = "surv_mo_60"
channels = [500]
dropouts = [0.1, 0.2, 0.5, 0.8]
weight_decays = [0.01, 0.001, 0]
lrs = [0.00005]
# best seems to be 0.8 dropout with 0.00005 lr

# Check the other survival lengths, e.g. with 0.00001, 0.00005, 0.0001 and 0.2, 0.5, 0.8 dropout.
for channel in channels:
    for dropout in dropouts:
        for weight_decay in weight_decays:
            for lr in lrs:
                command = f'python -m models.cnn --target {target} ' \
                          f'--table {table} --output-channel {channel} --weight-decay {weight_decay} --lr {lr}' \
                          f' --dropout {dropout}'
                print(f'executing command: {command}')
                check_output(command, shell=True)
