from subprocess import check_output

# Hyper-parameters
table = "bow_tuning"
target = "surv_mo_60"
max_tokens = [5000]  # 1000
# c_s = [0.1, 0.2, 0.3, 0.5]
# c_s = [0.6, 0.7, 0.8, 0.9, 1]
c_s = [1, 3, 5, 10, 100]
estimators = [50, 100, 200]

for max_token in max_tokens:
    classifier = "l2logreg"
    for c in c_s:
        command = f'python -m models.bow --target {target} --max-tokens {max_token} '\
                  f'--table {table}  --classifier {classifier} --l2logreg-c {c} --epochs 1'
        print(f'executing command: {command}')
        check_output(command, shell=True)

    classifier = "rf"
    # for estimator in estimators:
    #    command = f'python -m models.bow --target {target} --max-tokens {max_token} ' \
    #              f'--table {table}  --classifier {classifier} --rf-estimators {estimator} --epochs 1'
    #    print(f'executing command: {command}')
    #    check_output(command, shell=True)

