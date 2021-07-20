import argparse

import pandas as pd
import numpy as np

import bratlib.data as bd
from bratlib.calculators.entity_confusion_matrix import count_dataset
from bratlib.calculators.entity_confusion_matrix import count_file
# from bratlib.calculators.entity_agreement import measure_dataset
# from bratlib.calculators._utils import calculate_scores

# calculate precision, recall, and f1 scores
# measures = measure_dataset(gold_dataset, system_dataset)
# scores = calculate_scores(measures)


def cohen_kappa(matrix: pd.DataFrame):
    # returns Cohen's kappa statistic given a confusion matrix

    matrix = matrix.to_numpy()

    # compute the total number of agreements
    agreements = np.trace(matrix)

    # compute row and column totals of confusion matrix
    col_totals = matrix.sum(axis=0)
    row_totals = matrix.sum(axis=1)

    # compute sum of all matrix entries
    total = matrix.sum()

    # create expected frequency matrix
    ef_matrix = matrix
    for i in range(matrix.shape[0]):  # ith row
        for j in range(matrix.shape[1]):  # jth column
            ef = row_totals[i] * col_totals[j] / total
            ef_matrix[i, j] = ef

    # compute the total number of expected agreements
    e_agreements = np.trace(ef_matrix)

    # compute kappa statistic
    kappa = (agreements - e_agreements) / (total - e_agreements)

    return kappa


def cohen_kappa_file(anno_1: str, anno_2: str):
    # returns Cohen's kappa statistic given two .ann files

    file_1 = bd.BratFile.from_ann_path(anno_1)
    file_2 = bd.BratFile.from_ann_path(anno_2)

    # create confusion matrix
    matrix = count_file(file_1, file_2)

    # compute kappa statistic
    kappa = cohen_kappa(matrix)

    print("Cohen's kappa: ", kappa)


def cohen_kappa_dataset(anno_1: str, anno_2: str):
    # returns Cohen's kappa statistic given two datasets

    dataset_1 = bd.BratDataset.from_directory(anno_1)
    dataset_2 = bd.BratDataset.from_directory(anno_2)

    # create confusion matrix
    matrix = count_dataset(dataset_1, dataset_2)

    # compute kappa statistic
    kappa = cohen_kappa(matrix)

    print("Cohen's kappa: ", kappa)


# define directories
# gold_dataset = "./jessica/round_1/general"
# system_dataset = "./jenny/round_1/general"
#
# cohen_kappa_dataset(gold_dataset, system_dataset)

def main():
    parser = argparse.ArgumentParser(description="Calculate Cohen's kappa for two annotators")
    parser.add_argument('-m', '--mode', default='dir',
                        help='Compare two files (-m file) or two directories (-m dir)')
    parser.add_argument('anno_1', type=str,
                        help='Path to data for annotator 1')
    parser.add_argument('anno_2', type=str,
                        help='Path to data for annotator 2')
    args = parser.parse_args()

    if args.mode == 'dir':
        cohen_kappa_dataset(args.anno_1, args.anno_2)
    elif args.mode == 'file':
        cohen_kappa_file(args.anno_1, args.anno_2)
    else:
        raise ValueError("mode must be 'file' or 'dir'")

if __name__ == '__main__':
    main()