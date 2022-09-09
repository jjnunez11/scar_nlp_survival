from subprocess import check_output

# Hyper-parameters
table = "bert_tuning"
target = "surv_mo_60"
weight_decays = [0.1, 0.2, 0]
lrs = [0.0001, 0.00005, 0.00001]

for lr in lrs:
    for weight_decay in weight_decays:
        command = f'python -m models.bert --target {target} --batch-size 8 ' \
                  f'--table {table}  --lr {lr} --weight-decay {weight_decay}'
        print(f'executing command: {command}')
        check_output(command, shell=True)
