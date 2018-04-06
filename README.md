# AirlineRecommendation
Predict whether a customer will recommend an airline based on his/her review

## Goal
Based on the customer's review of the airline, predict whether a user will recommend this airline.

The train.csv contains the training data with customer's review and his/her recommendation status.
The test.csv only has the customer's review.
Our task is to predict whether the user in the test.csv will recommend the airline.

## Solution
We choose decision tree to make prediction.
Even though the built-in decision decision tree in skilearn is a convinient tool for this task, it cannot handle the missing data internally.
So we need to complete the training and testing data before using the built-in decision tree.

To improve the prediction accuracy, we implement the C4.5 algorithm by ourselves. The C4.5 decision tree can handle the missing data internally so it may produce better result.
