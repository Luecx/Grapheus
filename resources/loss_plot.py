import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import re
import glob

def read_log(file):
    df = pd.read_csv(file, usecols=[0,1,2], header=0)
    df.rename(columns={'training loss': file.split("/")[-2]}, inplace=True)
    return df


def retrieve_logs(folder):
    return glob.glob(os.path.join(folder, "**/loss*.csv"), recursive=True)


def read_logs(folder):
    logs = retrieve_logs(folder)
    return [read_log(log) for log in logs]


def do_plots(root_dirs):
    fig, ax = plt.subplots()

    for root_dir in root_dirs:
        datasets = read_logs(root_dir)
        for data in datasets:
            p = data.set_index("epoch").plot(
                ax=ax,
                lw=1,
                ylabel="loss",
            )




def main():
    parser = argparse.ArgumentParser(
        description="Generate plots of losses for an experiment run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "root_dirs",
        type=str,
        nargs="+",
        help="root directory(s) with loss*.csv",
    )

    args = parser.parse_args()
    do_plots(args.root_dirs)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
