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

def predict_overall_rating_only(train_data, test_data, test_result):
    """
    Use the overall rating only to build the decision tree
    """
    train_data = train_data[~np.isnan(train_data[:, 0])]
    add_indices_list = []
    for i, row in enumerate(train_data):
        if np.all(np.isnan(row[1:-1])):
            add_indices_list.append(i)
    train_data = train_data[add_indices_list]
    dtree = tree.DecisionTreeClassifier(criterion = "entropy")
    dtree.fit(train_data[:, 0][:, np.newaxis], train_data[:, -1])
    for i, row in enumerate(test_data):
        if np.all(np.isnan(row[1:])) and not np.isnan(row[0]):
            result = dtree.predict(np.array(row[0]).reshape(1, -1))
            test_result[i, 1] = result[0]

def predict_nan_only(test_data_raw, test_result):
    """
    If all information is nan, predict 0
    """
    for i, row in enumerate(test_data_raw):
        if np.all(np.isnan(row)):
            test_result[i, 1] = 0

def predict_overall_rating_missing(train_data, test_data, test_result):
    """
    Modify the result whose overall rating is missing
    """
    train_data = train_data[np.isnan(train_data[:, 0])][:, 1:]
    for row in train_data:
        for i in range(row.shape[0]):
            if np.isnan(row[i]):
                row[i] = -1
    # train the decision tree
    dtree = tree.DecisionTreeClassifier(criterion = "gini", min_samples_leaf=15)
    dtree.fit(train_data[:, :-1], train_data[:, -1])
    # modify the prediction
    for i, row in enumerate(test_data):
        if np.isnan(row[0]):
            row = row[1:]
            for j in range(row.shape[0]):
                if np.isnan(row[j]):
                    row[j] = -1
            result = dtree.predict(row.reshape(1,-1))[0]
            if result == 1:
                print("ha")
            test_result[i, 1] = result

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
    # delete the rows having no information
    delete_indices = []
    for i, row in enumerate(train_data_raw):
        if np.all(np.isnan(row[:-1])):
            delete_indices.append(i)
    train_data_raw = np.delete(train_data_raw, np.array(delete_indices), axis=0)
    # complete the train data and test data
    train_data, test_data = complete_train_and_test_data(\
        train_data_raw[:, :-1], test_data_raw[:, 1:])
    # train the decision tree
    dtree = tree.DecisionTreeClassifier(criterion = "gini", min_samples_leaf=15)
    dtree.fit(train_data, train_data_raw[:, -1])
    # make prediction
    prediction = dtree.predict(test_data)
    result = np.concatenate(\
        (test_data_raw[:, 0][:, np.newaxis], prediction[:, np.newaxis]), axis=1)
    # modify the result which only has overall rating
    predict_overall_rating_only(train_data_raw, test_data_raw[:, 1:], result)
    predict_overall_rating_missing(train_data_raw, test_data_raw[:, 1:], result)
    predict_nan_only(test_data_raw, result)
    # Save the result to file
    df = pd.DataFrame(result.astype(int))
    df.to_csv("qin_ruoxi.csv", header = ["id", "recommended"], index = False)
    print("finish")
