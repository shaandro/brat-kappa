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
    # ef_matrix = matrix.copy()
    # for i in range(matrix.shape[0]):  # ith row
    #     for j in range(matrix.shape[1]):  # jth column
    #         ef = row_totals[i] * col_totals[j] / total
    #         ef_matrix[i, j] = ef
    ef_matrix = np.outer(row_totals, col_totals) / total

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

    # print("Cohen's kappa: ", kappa)
    return kappa

categories = ["consult", "pharmacy", "discharge_summary", "general", "nursing", "physician"]
x = range(1,8)

jen_jes = pd.DataFrame(np.zeros((6, 13)), index=categories, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])

for i in categories[2:]:
    for j in range(1,14):
        anno_1 = "./jenny/round_" + str(j) + "/" + i
        anno_2 = "./jessica/round_" + str(j) + "/" + i
        jen_jes.loc[i].iloc[j-1] = cohen_kappa_dataset(anno_1, anno_2)

for j in range(1,4):
    anno_1 = "./jenny/round_" + str(j) + "/consult"
    anno_2 = "./jessica/round_" + str(j) + "/consult"
    jen_jes.loc["consult"].iloc[j-1] = cohen_kappa_dataset(anno_1, anno_2)

for j in range(1,9):
    anno_1 = "./jenny/round_" + str(j) + "/pharmacy"
    anno_2 = "./jessica/round_" + str(j) + "/pharmacy"
    jen_jes.loc["pharmacy"].iloc[j-1] = cohen_kappa_dataset(anno_1, anno_2)

print(jen_jes)

# had to remove "T85" from "adjudication/round_11/physician/54675_168205_571097_1.ann"
# and "adjudication/round_10/general/32511_166843_469604_1.ann"

jen_adj = pd.DataFrame(np.zeros((6, 13)), index=categories, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])

for i in categories[2:]:
    for j in range(1,14):
        anno_1 = "./jenny/round_" + str(j) + "/" + i
        anno_2 = "./adjudication/round_" + str(j) + "/" + i
        jen_adj.loc[i].iloc[j-1] = cohen_kappa_dataset(anno_1, anno_2)

for j in range(1,4):
    anno_1 = "./jenny/round_" + str(j) + "/consult"
    anno_2 = "./adjudication/round_" + str(j) + "/consult"
    jen_adj.loc["consult"].iloc[j-1] = cohen_kappa_dataset(anno_1, anno_2)

for j in range(1,9):
    anno_1 = "./jenny/round_" + str(j) + "/pharmacy"
    anno_2 = "./adjudication/round_" + str(j) + "/pharmacy"
    jen_adj.loc["pharmacy"].iloc[j-1] = cohen_kappa_dataset(anno_1, anno_2)

print(jen_adj)

# had to remove "T130" from "./adjudication/round_3/discharge_summary/200_02.ann"

jes_adj = pd.DataFrame(np.zeros((6, 13)), index=categories, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])

for i in categories[2:]:
    for j in range(1,14):
        anno_1 = "./jessica/round_" + str(j) + "/" + i
        anno_2 = "./adjudication/round_" + str(j) + "/" + i
        jes_adj.loc[i].iloc[j-1] = cohen_kappa_dataset(anno_1, anno_2)

for j in range(1,4):
    anno_1 = "./jessica/round_" + str(j) + "/consult"
    anno_2 = "./adjudication/round_" + str(j) + "/consult"
    jes_adj.loc["consult"].iloc[j-1] = cohen_kappa_dataset(anno_1, anno_2)

for j in range(1,9):
    anno_1 = "./jessica/round_" + str(j) + "/pharmacy"
    anno_2 = "./adjudication/round_" + str(j) + "/pharmacy"
    jes_adj.loc["pharmacy"].iloc[j-1] = cohen_kappa_dataset(anno_1, anno_2)

print(jes_adj)

jen_jes.to_csv('jenny_jessica.csv', index=True)
jen_adj.to_csv('jenny_adjudication.csv', index=True)
jes_adj.to_csv('jessica_adjudication.csv', index=True)