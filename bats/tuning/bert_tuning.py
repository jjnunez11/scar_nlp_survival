from subprocess import check_output

# Hyper-parameters
table = "bert_tuning"
target = "surv_mo_60"
weight_decays = [0, 0.1, 0.2]
lrs = [0.0001, 0.00005, 0.00001]

for weight_decay in weight_decays:
    for lr in lrs:
        command = f'python -m models.bert --target {target} ' \
                  f'--table {table}  --lr {lr}'
        print(f'executing command: {command}')
        check_output(command, shell=True)
