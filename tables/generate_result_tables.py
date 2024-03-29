import os
import pandas as pd
import re

pd.set_option("display.max_colwidth", 10000)


def generate_result_table(table):
    """
    Scans the results files and extracts the results that are part of the given table,
    and then writes the LaTeX string to a text file for copy pasting

    """
    f = open(r"result_tables\\" + table + ".txt", "w")

    table_df = pd.DataFrame([])

    for root, dirs, files in os.walk("../results/final_results/"):
        for file in files:
            if file[-11:] == 'results.csv' and not('old_results' in root):
                df = pd.read_csv(os.path.join(root, file))
                try:
                    df = df[df['Table'] == table]
                    df = df.drop_duplicates(['Model', 'Table Extra'], keep='last')
                    df = df[['Model', 'Table Extra', 'LaTeX String']]
                    table_df = table_df.append(df)
                    if len(df.index) > 0:
                        print(f'found results in {file}')
                except KeyError:
                    pass

    # Sort by Model, we'll have the order go BoW, CNN, LSTM, BERT. And then sort by the extra desc
    table_df['sort'] = 5
    table_df['sort'] = table_df['Model'].map({"BoW": 0, "CNN": 1, "LSTM": 2, "BERT": 3})
    table_df = table_df.sort_values(['sort', 'Table Extra'])

    latex_strings = table_df["LaTeX String"]

    horiz_sp = "\t"  # What to separate columns
    vert_sp = "\n"  # What to place at the end of the line to seperate rows vertically

    if len(latex_strings) > 0:
        s = latex_strings.to_string(header=False, index=False)
        s = re.sub(r"^\s*", "", s)
        s = re.sub(f"\n\s*", "", s)

        # Replace latex horizontal cell separator with specified
        s = re.sub(r"\s\&\s", horiz_sp, s)

        # Replace latex vertical cell separator with specified
        s = re.sub(r"\s\\\\", vert_sp, s)

        f.write(s + "\n")

    f.close()


if __name__ == "__main__":

    if True:  # all tables for paper
        # I didn't actually use this to put together the final paper results, fixed it now though
        generate_result_table('survival_dif_lengths')   # use this for all results, just copy pasted for the
        # different tables

    print("Printed table LaTeX string to file!")


