#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the unittest of DecisionTree class

**DEPRECATED**
This is the unittest for unweighted decision tree
@author: Ruoxi Qin
"""

import unittest
from decision_tree import DecisionTree
import numpy as np
from math import log
from collections import Counter

class DecisionTreeTest(unittest.TestCase):
    def test_divide_set_1_part(self):
        data = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        data = np.array(data)
        data = (data, np.ones(data.shape[0]))
        part1 = [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]
        correct = []
        correct.append(np.array(part1))
        result = DecisionTree.divide_set(0, [7], data)
        for i, v in enumerate(correct):
            self.assertTrue(np.all(v == result[i][0]))

    def test_divide_set_2_part(self):
        data = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        data = np.array(data)
        data = (data, np.ones(data.shape[0]))
        part1 = [[1, 2, 3],
                 [4, 5, 6]]
        part2 = [[7, 8, 9]]
        correct = []
        correct.append(np.array(part1))
        correct.append(np.array(part2))
        result = DecisionTree.divide_set(1, [5, 8], data)
        for i, v in enumerate(correct):
            self.assertTrue(np.all(v == result[i][0]))
    
    def test_divide_set_3_part(self):
        data = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        data = np.array(data)
        data = (data, np.ones(data.shape[0]))
        part1 = [[1, 2, 3]]
        part2 = [[4, 5, 6]]
        part3 = [[7, 8, 9]]
        correct = []
        correct.append(np.array(part1))
        correct.append(np.array(part2))
        correct.append(np.array(part3))
        result = DecisionTree.divide_set(2, [5, 8, 9], data)
        for i, v in enumerate(correct):
            self.assertTrue(np.all(v == result[i][0]))

    def test_claculate_entropy_even(self):
        data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        data = (data, np.ones(data.shape[0]))
        self.assertAlmostEqual(log(3), \
            DecisionTree.calculate_entropy(data))

    def test_claculate_entropy_empty(self):
        data = np.array([])
        data = (data, np.ones(data.shape[0]))
        self.assertAlmostEqual(0, DecisionTree.calculate_entropy(data))

    def test_claculate_entropy_pure(self):
        data = np.array([1, 1, 1])
        data = (data, np.ones(data.shape[0]))
        self.assertAlmostEqual(0, DecisionTree.calculate_entropy(data))

    def test_calculate_gain_ratio_best(self):
        class_tags_list = \
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        for i in range(len(class_tags_list)):
            class_tags_list[i] = \
                (class_tags_list[i], np.ones(class_tags_list[i].shape[0]))
        self.assertAlmostEqual(1, \
            DecisionTree.calculate_gain_ratio(class_tags_list))

    def test_calculate_gain_ratio_worst(self):
        class_tags_list = \
            [np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])]
        self.assertAlmostEqual(0, \
            DecisionTree.calculate_gain_ratio(class_tags_list))

    def test_calculate_gain_ratio_mix(self):
        class_tags_list = \
            [np.array([1, 2]), np.array([1, 1, 2, 2]), np.array([3, 3, 3])]
        correct = -(log(3) - 2 * log(2) / 3) / \
            (2 * log(2 / 9) / 9 + 4 * log(4 / 9) / 9 + log(1 / 3) / 3)
        self.assertAlmostEqual(correct, \
            DecisionTree.calculate_gain_ratio(class_tags_list))

    def test_calculate_gain_ratio_empty(self):
        class_tags_list = \
            [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3]), \
             np.array([])]
        self.assertAlmostEqual(1, \
            DecisionTree.calculate_gain_ratio(class_tags_list))

    def test_split(self):
        data = [[2, 0],
                [1, 0],
                [3, 1],
                [2, 0],
                [1, 0],
                [3, 1],
                [1, 0],
                [3, 1],
                [2, 0]]
        data = np.array(data)
        result, gain = DecisionTree.split(0, data)
        self.assertEqual(result, [2, 3])
        self.assertAlmostEqual(gain, 1)

    def test_split2(self):
        data = [[7, 2, 1],
                [8, 1, 0],
                [3, 3, 1],
                [6, 2, 1],
                [8, 1, 0],
                [1, 3, 1],
                [2, 1, 0],
                [5, 3, 1],
                [4, 2, 1]]
        data = np.array(data)
        result, gain = DecisionTree.split(1, data)
        self.assertEqual(result, [1, 3])
        self.assertAlmostEqual(gain, 1)

    def test_split_another_column(self):
        data = [[5, 1, 0],
                [9, 3, 1],
                [5, 3, 1],
                [5, 2, 0],
                [3, 2, 0],
                [5, 3, 1],
                [8, 2, 0],
                [4, 1, 0],
                [5, 1, 0]]
        data = np.array(data)
        result, gain = DecisionTree.split(1, data)
        self.assertEqual(result, [2, 3])
        self.assertAlmostEqual(gain, 1)

    def test_split_mix(self):
        data = [[5, 1, 0],
                [9, 1, 1],
                [5, 1, 0],
                [5, 2, 0],
                [3, 2, 0],
                [5, 2, 1],
                [8, 3, 1],
                [4, 3, 0],
                [5, 3, 0]]
        data = np.array(data)
        result, gain = DecisionTree.split(1, data)
        self.assertEqual(result, [3])

    def test_split_same(self):
        data = [[1, 0],
                [1, 0],
                [1, 0],
                [2, 0],
                [2, 0],
                [2, 0],
                [3, 0],
                [3, 0],
                [3, 0],
                [4, 0],
                [4, 0],
                [4, 0]]
        data = np.array(data)
        result, gain = DecisionTree.split(0, data)
        self.assertEqual(result, [4])
        self.assertAlmostEqual(gain, 0)

    def test_split_1_element(self):
        data = [[1, 0]]
        data = np.array(data)
        result, gain = DecisionTree.split(0, data)
        self.assertEqual(result, [1])
        self.assertAlmostEqual(gain, 0)

    def test_split_node_no_split(self):
        data = [[1, 0],
                [1, 0],
                [1, 0],
                [2, 0],
                [2, 0],
                [2, 0],
                [3, 0],
                [3, 0],
                [3, 0],
                [4, 0],
                [4, 0],
                [4, 0]]
        data = np.array(data)
        node = DecisionTree.split_node(data, set([0]))
        self.assertEqual(node.results, {0: 12})
        self.assertIsNone(node.children)

    def test_split_node_split(self):
        data = [[2, 0],
                [1, 0],
                [3, 1],
                [2, 0],
                [1, 0],
                [3, 1],
                [1, 0],
                [3, 1],
                [2, 0]]
        data = np.array(data)
        node = DecisionTree.split_node(data, set([0]))
        self.assertEqual(node.col, 0)
        self.assertEqual(node.values, [2, 3])
        self.assertEqual(node.count_list, [6, 3])
        self.assertEqual(node.children[0].col, -1)
        self.assertIsNone(node.children[0].children)
        self.assertEqual(node.children[0].results, {0: 6})
        self.assertEqual(node.children[1].col, -1)
        self.assertIsNone(node.children[1].children)
        self.assertEqual(node.children[1].results, {1: 3})

    def test_split_node_split2(self):
        data = [[7, 2, 1],
                [8, 1, 0],
                [3, 3, 1],
                [6, 2, 1],
                [8, 1, 0],
                [1, 3, 1],
                [2, 1, 0],
                [5, 3, 1],
                [4, 2, 1]]
        data = np.array(data)
        node = DecisionTree.split_node(data, set([0, 1]))
        self.assertEqual(node.col, 1)
        self.assertEqual(node.values, [1, 3])
        self.assertIsNone(node.results)
        self.assertEqual(node.count_list, [3, 6])
        self.assertEqual(node.children[0].col, -1)
        self.assertIsNone(node.children[0].children)
        self.assertEqual(node.children[0].results, {0: 3})
        self.assertEqual(node.children[1].col, -1)
        self.assertIsNone(node.children[1].children)
        self.assertEqual(node.children[1].results, {1: 6})

    def test_init(self):
        data = [[7, 2, 1],
                [8, 1, 0],
                [3, 3, 1],
                [6, 2, 1],
                [8, 1, 0],
                [1, 3, 1],
                [2, 1, 0],
                [5, 3, 1],
                [4, 2, 1]]
        data = np.array(data)
        tree = DecisionTree(data)
        node = tree._root
        self.assertEqual(node.col, 1)
        self.assertEqual(node.values, [1, 3])
        self.assertIsNone(node.results)
        self.assertEqual(node.count_list, [3, 6])
        self.assertEqual(node.children[0].col, -1)
        self.assertIsNone(node.children[0].children)
        self.assertEqual(node.children[0].results, {0: 3})
        self.assertEqual(node.children[1].col, -1)
        self.assertIsNone(node.children[1].children)
        self.assertEqual(node.children[1].results, {1: 6})

    def test_predict_1_layer(self):
        data = [[7, 2, 1],
                [8, 1, 0],
                [3, 3, 1],
                [6, 2, 1],
                [8, 1, 0],
                [1, 3, 1],
                [2, 1, 0],
                [5, 3, 1],
                [4, 2, 1]]
        data = np.array(data)
        tree = DecisionTree(data)
        predict_data = [3, 1]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(0, tree.predict(predict_data))
        predict_data = [7, 2]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(1, tree.predict(predict_data))
        predict_data = [8, 3]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(1, tree.predict(predict_data))

    def test_predict_1_layer_with_missing_data(self):
        data = [[7, 2, 1],
                [8, 1, 0],
                [3, 3, 1],
                [6, 2, 1],
                [8, 1, 0],
                [1, 3, 1],
                [2, 1, 0],
                [5, 3, 1],
                [4, 2, 1]]
        data = np.array(data)
        tree = DecisionTree(data)
        predict_data = [None, 1]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(0, tree.predict(predict_data))
        predict_data = [7, None]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(1, tree.predict(predict_data))
        predict_data = [8, None]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(1, tree.predict(predict_data))

    def test_predict_2_layer(self):
        data = [[2, 1, 0],
                [4, 1, 1],
                [4, 1, 1],
                [2, 2, 1],
                [3, 2, 0],
                [3, 2, 0],
                [3, 3, 1],
                [4, 3, 0],
                [4, 3, 0],
                [4, 1, 1],
                [2, 2, 1],
                [3, 2, 0],
                [3, 3, 1],
                [4, 3, 0],
                [2, 1, 0]]
        data = np.array(data)
        tree = DecisionTree(data)
        predict_data = [4, 1]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(1, tree.predict(predict_data))
        predict_data = [1, 3]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(1, tree.predict(predict_data))
        predict_data = [2, 1]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(0, tree.predict(predict_data))

    def test_predict_2_layer_with_missing(self):
        data = [[2, 1, 0],
                [4, 1, 1],
                [4, 1, 1],
                [2, 2, 1],
                [3, 2, 0],
                [3, 2, 0],
                [3, 3, 1],
                [4, 3, 0],
                [4, 3, 0],
                [4, 1, 1],
                [2, 2, 1],
                [3, 2, 0],
                [3, 3, 1],
                [4, 3, 0],
                [2, 1, 0]]
        data = np.array(data)
        tree = DecisionTree(data)
        predict_data = [None, 1]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(1, tree.predict(predict_data))
        predict_data = [None, 3]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(0, tree.predict(predict_data))
        predict_data = [2, None]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(0, tree.predict(predict_data))
        predict_data = [3, None]
        predict_data = np.array(predict_data)
        self.assertAlmostEqual(0, tree.predict(predict_data))

    def test_counter_entropy(self):
        counter = Counter({1:8, 2:2})
        correct = -(0.8 * log(0.8) + 0.2 * log(0.2))
        self.assertAlmostEqual(correct, DecisionTree.counter_entropy(counter))

    def test_counter_entropy_0(self):
        counter = Counter({1:8})
        self.assertAlmostEqual(0, DecisionTree.counter_entropy(counter))


if __name__ == "__main__":
    unittest.main()
