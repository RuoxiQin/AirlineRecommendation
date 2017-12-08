#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@project CS565 Project 2
@author Ruoxi Qin
@Interpreter version Python3.6

Train the decision tree and get the prediction result
'''

import numpy as np
import pandas as pd
from math import log
from math import inf
from collections import Counter
from decision_tree import DecisionTree

def train_decision_tree(training_set, complete_method):
    """
    Train the decision tree

    :param training_set: the path of the training data file -- String
    :param complete_method: The method to complete the missing data -- Function
    :return: the trained decision tree -- DecisionTree
    """
    # read the csv file and get the training data
    df = pd.read_csv(training_set)
    column_names = ["overall_rating", "seat_comfort_rating", \
    "cabin_staff_rating", "food_beverages_rating", \
    "inflight_entertainment_rating", "value_money_rating", "recommended"]
    data = df.as_matrix(column_names)
    # Complete the data using given method
    data = complete_method(data)
    # train the decision tree
    return DecisionTree(data)

def make_prediction(tree, test_set):
    """
    Use the decision tree to make the prediction of the test_set

    :param tree: Decision tree -- DecisionTree
    :param test_set: The path of the test_set file -- String
    :return: The prediction matrix -- numpy.ndarray
    """
    # read the csv file and get the training data
    df = pd.read_csv(test_set)
    column_names = ["id", "overall_rating", "seat_comfort_rating", \
    "cabin_staff_rating", "food_beverages_rating", \
    "inflight_entertainment_rating", "value_money_rating"]
    data = df.as_matrix(column_names)
    # predict each result and return the prediction matrix
    zero_vector = np.zeros(data.shape[0])
    result = np.c_[data[:, 0], zero_vector]
    for i, row in enumerate(data):
        result[i, 1] = tree.predict(row[1:], np.isnan)
    return result.astype(int)

def guess_on_overall_average(data):
    """
    Complete the missing data with the rounded average of each columns

    :param data: data matrix to complete -- Numpy.array
    :return: Completed data matrix -- Numpy.array
    """
    # wash out the data without overall_rating
    data = data[~np.isnan(data[:, 0])]
    # calculate the average rate of each column giving overall_rating
    average_ratings = []
    for rate in range(1, 11):
        average_ratings.append(np.around(np.nanmean(\
            data[data[:, 0] == rate, 1 : -1], 0)))
    # replace the missing data according to its overall_rating and 
    # average_ratings
    for row in data:
        if not np.isnan(row[0]):
            for j in range(1, 6):
                if np.isnan(row[j]):
                    row[j] = average_ratings[int(row[0]) - 1][j - 1]
    return data

def wash_out_all_nan(data):
    """
    Wash out all columns having missing data

    :param data: data matrix to complete -- Numpy.array
    :return: Completed data matrix -- Numpy.array
    """
    # wash out the data with nan
    return data[~np.isnan(np.sum(data, 1))]

def no_modify(data):
    """
    Don't do anything on data

    :param data: data matrix to complete -- Numpy.array
    :return: original data matrix -- Numpy.array
    """
    # wash out the data with nan
    return data


if __name__ == "__main__":
    # Train the decision tree
    train_data_path = "train.csv"
    tree = train_decision_tree(train_data_path, no_modify)
    # Prune the tree
    tree.prune(0.02)
    # make prediction using the decision tree
    test_data_path = "test.csv"
    prediction_result = make_prediction(tree, test_data_path)
    # save the prediction to csv file
    df = pd.DataFrame(prediction_result)
    df.to_csv("qin_ruoxi.csv", header = ["id", "recommended"], index = False)
    print("finish")
