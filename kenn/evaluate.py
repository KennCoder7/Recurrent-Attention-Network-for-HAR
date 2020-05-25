import numpy as np


# def evaluate_(true, pred):
#     true_ = true[:, 1:-1]
#     pred_ = pred[:, :-1]
#
#     n_true = 0
#     for i in range(true_.shape[0]):
#         for j in range(true_.shape[1]):
#             if j > pred_.shape[1]:
#                 pass
#             else:
#                 if true_[i][j] == pred_[i][j]:
#                     n_true += 1
#     return n_true / (true_.shape[0] * true_.shape[1])


def evaluate_(true, pred):
    true_ = true[:, 1:3]
    pred_ = pred[:, 0:2]

    n_true = np.empty(0)
    for i in range(true_.shape[0]):
        n_true = np.append(n_true, np.equal(true_[i], pred_[i]))
    return np.mean(n_true)


def evaluate(true, pred):
    true_ = true[:, 1:3]
    pred_ = pred[:, 0:2]

    n_true = []
    for i in range(true_.shape[0]):
        n_true = np.append(n_true, (true_[i] == pred_[i]).all())
    return np.mean(n_true)
