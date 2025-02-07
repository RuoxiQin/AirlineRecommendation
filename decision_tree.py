#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@project Airline Recommendation
@author Ruoxi Qin
@Interpreter version Python3.6

The C4.5 decision tree
'''

import numpy as np
from math import log
from math import inf
from collections import Counter

class _DecisionTreeNode:
    def __init__(\
        self, col = -1, values = None, results = None, children = None, \
        count_list = None):
        """
        The tree node of the decision tree.

        :param col: The column index of the split criterion -- Int
        :param values: The <= split point -- List[Int]
        :param results: Counter of the result in this node. Is None if the node is not the leaf. -- collections.Counter
        :param children: Children nodes -- List[_DecisionTreeNode]
        :param count_list: The count of training data in each children -- 
        List[Int]
        """
        self.col = col
        self.values = values
        self.results = results
        self.children = children
        self.count_list = count_list

class DecisionTree:
    def __init__(self, data):
        """
        Use the data to generate a decision tree. 

        :param data: The data matrix of training data. The last column is the 
        classification tag. -- numpy.ndarray
        """
        # Build decision tree
        possible_column = set(range(data.shape[1] - 1))
        self._root = DecisionTree.split_node(data, possible_column)

    def split_node(data, possible_column):
        """
        Split and build the decision tree according to column in
        possible_column and return the root

        :param data: Data set of the node -- numpy.ndarray
        :param possible_column: Avaliable column index -- Set(Int) 
        :return: The root of this decision tree -- _DecisionTreeNode
        """
        max_gain = -np.inf
        for col in possible_column:
            split, gain = DecisionTree.split(col, data)
            if gain > max_gain:
                max_gain = gain
                max_split = split
                max_col = col
        if max_gain > 0.00001:
            possible_column.remove(max_col)
            node = _DecisionTreeNode(max_col, max_split, None, [], [])
            for data_set in DecisionTree.divide_set(max_col, max_split, data):
                node.children.append(\
                    DecisionTree.split_node(data_set, possible_column))
                node.count_list.append(len(data_set))
            # After building the subtree, add back the used column for the 
            # brother branches
            possible_column.add(max_col)
        else:
            result = Counter()
            for i, v in enumerate(data[0][:, -1]):
                result.update({v: data[1][i]})
            node = _DecisionTreeNode(results = result)
        return node

    def divide_set(col, vals, data, is_empty=np.isnan):
        """
        Divide the data according to the thresholds in vals of column col.
        Does not allow empty subset!

        :param col: Column index -- Int
        :param vals: Dividing thresholds. Divide the data <= thresholds in to
        one subnode -- List[Int]
        :param data: data to split and its weight -- 
        (numpy.ndarray, numpy.ndarray)
        :return: List of splitted data and weight -- 
        List[(numpy.ndarray, numpy.ndarray)]
        """
        weights = data[1]
        data = data[0]
        is_empty_list = is_empty(data[:, col])
        missing_data = data[is_empty_list]
        missing_data_weight = weights[is_empty_list]
        missing_data_weight_sum = np.sum(missing_data_weight)
        data = data[~is_empty_list]
        data_weight = weights[~is_empty_list]
        nonmissing_data_weight_sum = np.sum(data_weight)
        data_list = []
        upper_bound = -np.inf
        for bound in vals:
            lower_bound = upper_bound
            upper_bound = bound
            choose = (data[:, col] > lower_bound) & \
                (data[:, col] <= upper_bound)
            one_data = np.concatenate((data[choose], missing_data))
            one_weight_array = weights[choose]
            frac = np.sum(one_weight_array) / nonmissing_data_weight_sum
            one_weight = \
                np.concatenate((one_weight_array, frac * missing_data_weight))
            data_list.append((one_data, one_weight))
        return data_list

    def calculate_entropy(class_tags):
        """
        Calculate the entropy of a given class_tags list

        :param class_tags: Classification tags list with weight -- 
        (numpy.ndarray, numpy.ndarray)
        :return: The entropy of this list -- float
        """
        if len(class_tags[0]) == 0:
            return 0
        total_weight = np.sum(class_tags[1])
        entropy = 0
        counter = Counter()
        for i in range(class_tags[0].shape[0]):
            counter.update({class_tags[0][i]: class_tags[1][i]})
        for count in counter.values():
            fraction = count / total_weight
            entropy += fraction * log(fraction)
        return -entropy

    def counter_entropy(counter):
        """
        Calculate the entropy using the counter

        :param result: the counter (or result) of a node -- Counter
        :return: the entropy of this node -- float
        """
        if len(counter)  == 0:
            return 0
        total_count = sum(counter.values())
        entropy = 0
        for count in counter.values():
            if count > 0:
                fraction = count / total_count
                entropy += fraction * log(fraction)
        return -entropy

    def calculate_gain_ratio(class_tags_list, missing_data_num = 0):
        """
        Calculate the gain ratio of a given split

        :param class_tags_list: The classification tags of each sub nodes -- 
        List[(numpy.ndarray, numpy.ndarray)]
        :param missing_data_num: number of missing data -- Int
        :return: Gain ratio of this split -- float
        """
        if len(class_tags_list) <= 1:
            return 0
        parent_tags = np.concatenate([pair[0] for pair in class_tags_list])
        total_unmissing_weight = 0
        for pair in class_tags_list:
            total_unmissing_weight += np.sum(pair[1])
        complete_fraction = \
            1 - missing_data_num / (total_unmissing_weight + missing_data_num)
        children_entropy = 0
        split_info = 0
        for subnode_tag in class_tags_list:
            subnode_weight_sum = np.sum(subnode_tag[1])
            children_entropy += subnode_weight_sum * \
                DecisionTree.calculate_entropy(subnode_tag)
            split_fraction = \
                subnode_weight_sum / (total_unmissing_weight + missing_data_num)
            if split_fraction > 0:
                split_info += split_fraction * log(split_fraction)
        if missing_data_num > 0:
            split_fraction = \
                missing_data_num / (total_unmissing_weight + missing_data_num)
            split_info += split_fraction * log(split_fraction)
        delta = DecisionTree.calculate_entropy(parent_tags) - \
            children_entropy / total_unmissing_weight
        return complete_fraction * delta / (-split_info)

    def split(col, data, is_empty=np.isnan):
        """
        Use dynamic programming to Find the best split of the data into best
        number of parts

        :param col: Column index to split the data -- Int
        :param data: Data matrix with the last column designating the 
        classification tag. The second part of data is the weight -- 
        (numpy.ndarray, numpy.ndarray)
        :param is_empty: the function to test the empty variable -- function
        :return: Split point with number <= split point being in 1 part -- 
        List[Int]
        """
        data_weight = data[1]
        data = data[0]
        different_position = []
        is_empty_list = is_empty(data[:, col])
        missing_data = data[is_empty_list]
        missing_data_weight = data_weight[is_empty_list]
        missing_data_weight_sum = np.sum(missing_data_weight)
        data = data[~is_empty_list]
        data_weight = data_weight[~is_empty_list]
        data = data[data[:, col].argsort()]
        data_weight = data_weight[data[:, col].argsort()]
        # Find how many different split points are there
        for i in range(len(data) - 1):
            if data[i, col] != data[i + 1, col]:
                different_position.append(i)                
        different_position.append(len(data) - 1)
        k = len(different_position)
        # Pre-computation: unit_table
        unit_table = \
            np.zeros((len(different_position), len(different_position)))
        start_point_position = [0]
        for i in range(len(different_position) - 1):
            start_point_position.append(different_position[i] + 1)
        for i in range(len(start_point_position)):
            entropy = 0
            main_counter = Counter()
            sum_log = 0
            start_point = start_point_position[i]
            for j in range(i, len(different_position)):
                new_counter = Counter()
                for _index, _tag in \
                    enumerate(data[start_point: different_position[j] + 1, -1]):
                    new_counter.update({_tag: data_weight[i]})
                for value, count in new_counter.items():
                    if value in main_counter:
                        modified_count = main_counter[value]
                        sum_log += -modified_count * log(modified_count) + \
                            (modified_count + count) * \
                            log(modified_count + count)
                    else:
                        sum_log += count * log(count)
                main_counter.update(new_counter)
                unit_table[i, j] = -(sum_log / sum(main_counter.values()) - \
                    log(sum(main_counter.values())))
                start_point = different_position[j] + 1
        # Start dynamic programming
        OPT = np.zeros((k, len(different_position)))
        split_position_matrix = \
            [[0] * k for i in range(len(different_position))]
        for j in range(len(different_position)):
            split_position_matrix[0][j] = j
        for i in range(1, k):
            split_position_matrix[i][i] = i - 1
        for j in range(len(different_position)):
            OPT[0, j] = unit_table[0, j]
        for i in range(1, k):
            for j in range(i + 1, len(different_position)):
                min_entropy = np.inf
                for start in range(i, j + 1):
                    new_entropy = OPT[i - 1, start - 1] + unit_table[start, j]
                    if new_entropy < min_entropy:
                        min_entropy = new_entropy
                        split_position_matrix[i][j] = start - 1
                OPT[i, j] = min_entropy
        # Use the split_info to find the best split
        max_gain_ratio = -np.inf
        for i in range(k):
            split_point = []
            position = len(different_position) - 1
            for step in range(i, -1, -1):
                split_point.append(different_position[position])
                position = split_position_matrix[step][position]
            split_point.append(-1)
            split_point.reverse()
            class_tag_list = []
            for _i in range(len(split_point) - 1):
                new_data_list = \
                    data[split_point[_i] + 1 : split_point[_i + 1] + 1, -1]
                new_weight_list = \
                    data_weight[split_point[_i] + 1 : split_point[_i + 1] + 1]
                class_tag_list.append((new_data_list, new_weight_list))
            gain_ratio = \
                DecisionTree.calculate_gain_ratio(\
                class_tag_list, missing_data_weight_sum)
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                best_split = split_point
        best_split.pop(0)  # pop the first -1
        return [data[x, col] for x in best_split], max_gain_ratio

    def predict(self, data, is_empty=lambda x: x is None):
        """
        Predict the tag of the given data according to the decision tree
        The default missing data is represented as None

        :param data: Data column to predict -- numpy.ndarray
        :param is_empty: The function to denote the missing data -- 
        Default to be testing None
        :return: tag -- Same type as the last column of the training data
        """
        def _get_result(observation, node, total_weight=True):
            if node.results is not None:
                return node.results
            else:
                v = observation[node.col]
                if is_empty(v):
                    results_list = [_get_result(observation, child, \
                    total_weight) for child in node.children]
                    if total_weight:
                        total_count = sum(node.count_list)
                        weight_list = [count / total_count \
                            for count in node.count_list]
                    else:
                        total_count = 0
                        weight_list = []
                        for one_result in results_list:
                            sum_one_result = sum(one_result.values())
                            total_count += sum_one_result
                            weight_list.append(sum_one_result)
                        weight_list = [w / total_count for w in weight_list]
                    final_result = Counter()
                    for i, result in enumerate(results_list):
                        for k, v in result.items():
                            final_result.update({k: v * weight_list[i]})
                    return final_result
                else:
                    for i, upper_bound in enumerate(node.values):
                        if observation[node.col] <= upper_bound:
                            return _get_result(observation, node.children[i], \
                                total_weight)
                    # if the data is beyond the training set upperbound, it
                    # actually belongs to the last children
                    return _get_result(observation, node.children[-1], \
                        total_weight)
        result_counter = _get_result(data, self._root, False)
        max_count = 0
        for k, v in result_counter.items():
            if v > max_count:
                max_count = v
                tag = k
        return tag

    def prune(self, min_gain):
        """
        Prune the decision tree. Merge leaves which contributes the gain less
        than min_gain.

        :param min_gain: The minimum gain threshold to prune the leaves -- float
        """
        def _prune(node):
            i = 0
            while i + 1 < len(node.children):
                # If the child of the node is not leaf, prune the
                # that child first
                if node.children[i].results is None:
                    _prune(node.children[i])
                if node.children[i + 1].results is None:
                    _prune(node.children[i + 1])
                # If they are leaves now, we can prune this pair of nodes
                if node.children[i].results is not None and \
                    node.children[i + 1].results is not None:
                    combined_result = Counter()
                    combined_result.update(node.children[i].results)
                    combined_result.update(node.children[i + 1].results)
                    sum_combined_result = sum(combined_result.values())
                    sum_result_i = sum(node.children[i].results.values())
                    sum_result_i1 = sum(node.children[i+1].results.values())
                    # If the gain is smaller than the threshold, merge them
                    delta = DecisionTree.counter_entropy(combined_result) - \
                        (DecisionTree.counter_entropy(\
                        node.children[i].results) * sum_result_i +
                        DecisionTree.counter_entropy(\
                        node.children[i+1].results) * sum_result_i1) / \
                        sum_combined_result
                    # print(delta)
                    if delta < min_gain:
                        if len(node.children) > 2:
                            # if there are other children, just change this
                            # pair of children
                            node.count_list[i] += node.count_list[i + 1]
                            del node.count_list[i + 1]
                            del node.values[i]
                            node.children[i] = _DecisionTreeNode(\
                                results = combined_result)
                            del node.children[i + 1]
                            # Because the children decreased by 1, we don't 
                            # need to increase i
                        else:
                            # if this is the last pair of children, this node
                            # becomes leaf
                            node.col = None
                            node.values = None
                            node.results = combined_result
                            node.children = None
                            node.count_list = None
                            break
                    else:
                        # Otherwise prune the next pair of nodes
                        i += 1
                else:
                    # Otherwise prune the next pair of nodes
                    i += 1
        _prune(self._root)
