# Logistic Regression

This page demostrates some sample code for logistic regression. Two Python files has been shared.

`logistic_regression.py` is written from scratch without any Machine Learning libraries such as `scikit-learn`.

`logistic_regression_sklearn.py` uses the capabilities of `scikit-learn` tools.

## Results

### Gradient Descend

`linear_regression.py` manages to decrease the cost and it converges as seen below.

<img src="https://user-images.githubusercontent.com/22200109/210359705-eb296aff-1227-4b6e-9de0-c7f343c12ee8.png" width="500">

### Accuracy

Using the same train and and test sets, the accuracy scores of both version:


|   | Accuracy |
| ------------- | ------------- |
| `logistic_regression.py`  | 0.7846  |
| `logistic_regression_sklearn.py`  | 0.7846  |

In the end, we get the same result.
