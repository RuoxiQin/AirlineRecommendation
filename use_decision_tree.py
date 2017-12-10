#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@project CS565 Project 2
@author Ruoxi Qin
@Interpreter version Python3.6

Use the decision tree
'''

import numpy as np
import pandas as pd
from math import log
from math import inf
from sklearn import tree
from collections import Counter
import graphviz

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

def complete_train_and_test_data(train_data, test_data):
    """
    complete the train data and the test data according to the first column
    if the first column is empty, it is set to be 0
    """
    all_data = np.concatenate((train_data, test_data), axis=0)
    # replace all the missing value in the first column as 0
    for row in all_data:
        if np.isnan(row[0]):
            row[0] = 0
    # calculate the average rate of each column giving overall_rating
    average_ratings = []
    for rate in range(11):
        average_ratings.append(np.around(np.nanmean(\
            all_data[all_data[:, 0] == rate, 1 :], axis=0)))
    # replace the missing data according to its overall_rating and 
    # average_ratings
    for row in all_data:
        for j in range(1, all_data.shape[1]):
            if np.isnan(row[j]):
                row[j] = average_ratings[int(row[0])][j - 1]
    return all_data[:train_data.shape[0]], all_data[train_data.shape[0]:]

def get_airline_data(train_data, test_data):
    """
    Transfer string airline name to id
    """
    all_data = np.concatenate((train_data, test_data))
    id_set = set(all_data[:, 0])
    id_dict = {}
    for id_, name in enumerate(id_set):
        id_dict[name] = id_
    for row in all_data:
        row[0] = id_dict[row[0]]
    return all_data[:train_data.shape[0]], all_data[train_data.shape[0]:]

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
    train_file = "train.csv"
    test_file = "test.csv"
    # read the train csv file and get the training data
    df = pd.read_csv(train_file)
    column_names = ["overall_rating", "seat_comfort_rating", \
    "cabin_staff_rating", "food_beverages_rating", \
    "inflight_entertainment_rating", "value_money_rating", "recommended"]
    train_data_raw = df.as_matrix(column_names)
    # read the test csv file and get the test data
    df = pd.read_csv(test_file)
    column_names = ["id", "overall_rating", "seat_comfort_rating", \
    "cabin_staff_rating", "food_beverages_rating", \
    "inflight_entertainment_rating", "value_money_rating"]
    test_data_raw = df.as_matrix(column_names)
    # complete the train data and test data
    train_data, test_data = complete_train_and_test_data(\
        train_data_raw[:, :-1], test_data_raw[:, 1:])

    '''
    # get the airline name
    df = pd.read_csv(train_file)
    airline_train = df.as_matrix(["airline_name"])
    df = pd.read_csv(test_file)
    airline_test = df.as_matrix(["airline_name"])
    airline_train_id, airline_test_id = \
        get_airline_data(airline_train, airline_test)
    train_data = np.concatenate((train_data, airline_train_id), axis=1)
    test_data = np.concatenate((test_data, airline_test_id), axis=1)
    '''

    # train the decision tree
    dtree = tree.DecisionTreeClassifier(criterion = "gini", min_samples_leaf=15)
    dtree.fit(train_data, train_data_raw[:, -1])
    # print the tree
    dot_data = tree.export_graphviz(dtree, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    '''
    # train the naive bayes
    gnb = GaussianNB()
    gnb.fit(train_data_raw[:-1], train_data_raw[:, -1])
    '''
    # make prediction
    prediction = dtree.predict(test_data)
    result = np.concatenate(\
        (test_data_raw[:, 0][:, np.newaxis], prediction[:, np.newaxis]), axis=1)
    df = pd.DataFrame(result.astype(int))
    df.to_csv("qin_ruoxi.csv", header = ["id", "recommended"], index = False)
    print("finish")
