from subprocess import check_output

# Hyper-parameters
table = "bow_tuning"
target = "surv_mo_60"
max_tokens = [1000, 5000, 10000]
c_s = [0.1, 0.2, 0.3, 0.5]
estimators = [50, 100, 200]

for max_token in max_tokens:
    classifier = "l2logreg"
    for c in c_s:
        command = f'python -m models.bow --target {target} ' \
                  f'--table {table}  --classifier {classifier} --l2logreg-c {c}'
        print(f'executing command: {command}')
        check_output(command, shell=True)

    classifier = "rf"
    for estimator in estimators:
        command = f'python -m models.bow --target {target} ' \
                  f'--table {table}  --classifier {classifier} --rf-estimators {estimator}'
        print(f'executing command: {command}')
        check_output(command, shell=True)

