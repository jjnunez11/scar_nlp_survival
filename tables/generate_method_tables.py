import os
import pandas as pd
from tables.table_globals import METHOD_TABLES_DIR, LABEL_DIR, PT_DATA_FILE

"""
Table generation. Adapted from thesis but to output Google doc compliant tables, not LaTeX
"""





def generate_dspln_label_table():
    f = open(os.path.join(METHOD_TABLES_DIR, "dspln_label" + ".txt"), "w")

    f.write("Psychiatry & " + get_label_balance("dsplnic_PSYCHIATRY_60"))
    f.write("\n")
    f.write("Counselling & " + get_label_balance("dsplnic_SOCIALWORK_60"))

    f.close()


def generate_need_label_table():
    f = open(os.path.join(METHOD_TABLES_DIR, "need_label" + ".txt"), "w")

    f.write("Emotional & 1 & " + get_label_balance("need_emots_1") + "\n")
    f.write("Emotional & 2 & " + get_label_balance("need_emots_2") + "\n")
    f.write("Emotional & 3 & " + get_label_balance("need_emots_3") + "\n")
    f.write("Emotional & 4 & " + get_label_balance("need_emots_4") + "\n")
    f.write("Emotional & 5 & " + get_label_balance("need_emots_5") + "\n")
    f.write("Informational & 1 & " + get_label_balance("need_infos_1") + "\n")
    f.write("Informational & 2 & " + get_label_balance("need_infos_2") + "\n")
    f.write("Informational & 3 & " + get_label_balance("need_infos_3") + "\n")
    f.write("Informational & 4 & " + get_label_balance("need_infos_4") + "\n")

    f.close()


def generate_survival_label_table():
    f = open(os.path.join(METHOD_TABLES_DIR, "survival_label" + ".txt"), "w")

    f.write("6 & " + get_label_balance("survic_mo_6", True) + "\n")
    f.write("36 & " + get_label_balance("survic_mo_36", True) + "\n")
    f.write("60 & " + get_label_balance("survic_mo_60", True) + "\n")

    f.close()


def generate_pt_demo_table():
    filename = PT_DATA_FILE

    df = pd.read_csv(filename)
    total_n = len(df.index)
    f = open(os.path.join(METHOD_TABLES_DIR, "pt_demo" + ".txt"), "w")

    horiz_sp = "\t"  # What to separate columns
    vert_sp = "\n"  # What to place at the end of the line to seperate rows vertically

    def get_label_balance(label_name, with_total=False):
        root = LABEL_DIR
        file = os.path.join(root, "y_" + label_name + ".csv")
        df = pd.read_csv(file)
        n_one = len(df[df['label'] == 1])
        n_zero = len(df[df['label'] == 0])
        n_pts = n_one + n_zero

        perc_one = round(100 * n_one / n_pts, 1)
        perc_zero = round(100 * n_zero / n_pts, 1)

        if with_total:
            label_balance_str = f"{n_zero} ({perc_zero}){horiz_sp}{n_one} ({perc_one}){horiz_sp}{n_pts}{vert_sp}"
        else:
            label_balance_str = f"{n_zero} ({perc_zero}){horiz_sp}({perc_one}){vert_sp}"

        return label_balance_str

    def get_n_perc(df, col, val, total):
        if val == "":
            n = df[col].isnull().values.ravel().sum()
        else:
            n = len(df[df[col] == val].index)

        perc = round(100 * n / total, 1)

        return f'{n}{horiz_sp}{perc}{vert_sp}'

    def get_mean_std(df, col):
        mean = round(df[col].mean(), 1)
        std = round(df[col].std(), 1)

        return f'{mean}{horiz_sp}{std}{vert_sp}'

    f.write(f"{horiz_sp}n{horiz_sp}%{vert_sp}")
    f.write(f"Total{horiz_sp}{total_n}{horiz_sp}100{vert_sp}")
    f.write(f"Female{horiz_sp}"+ get_n_perc(df,"sex","F",total_n))
    f.write(f"Stage I{horiz_sp}" + get_n_perc(df, "stage", "I", total_n))
    f.write(f"Stage II{horiz_sp}" + get_n_perc(df, "stage", "II", total_n))
    f.write(f"Stage III{horiz_sp}" + get_n_perc(df, "stage", "III", total_n))
    f.write(f"Stage IV{horiz_sp}" + get_n_perc(df, "stage", "IV", total_n))
    f.write(f"Unknown Stage{horiz_sp}" + get_n_perc(df, "stage", "", total_n))
    f.write(vert_sp)
    f.write(f"{horiz_sp}Mean{horiz_sp}Standard Deviation{vert_sp}")
    f.write(f'Age at Diagnosis{horiz_sp}' + get_mean_std(df, "age_at_diagnosis"))
    f.write(f'Months Survived since Diagnosis{horiz_sp}' + get_mean_std(df, "mo_survived"))
    f.write(f'Months Survived since Document{horiz_sp}' + get_mean_std(df, "mo_survived_since_initial_consult"))
    f.write(vert_sp)
    f.write(f"{horiz_sp}Did not Survive (%){horiz_sp}Survived (%){vert_sp}")
    f.write(f"6 Months{horiz_sp}" + get_label_balance("surv_mo_6", False))
    f.write(f"36 Months{horiz_sp} " + get_label_balance("surv_mo_36", False))
    f.write(f"60 Months{horiz_sp} " + get_label_balance("surv_mo_60", False))

    f.close()

    print(len(df.index))


if __name__ == "__main__":
    # generate_dspln_label_table()
    # generate_need_label_table()
    # generate_survival_label_table()
    generate_pt_demo_table()

    print("Table generation complete!")
