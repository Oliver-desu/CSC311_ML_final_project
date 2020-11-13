from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    num_data = len(data["user_id"])
    for index in range(num_data):
        answer = data["is_correct"][index]
        j = data["question_id"][index]
        i = data["user_id"][index]
        d = theta[i] - beta[j]
        if answer == 1:
            log_lklihood += d - np.log(1+np.exp(d))
        elif answer == 0:
            log_lklihood += - np.log(1+np.exp(d))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    d_theta = np.zeros(theta.shape)
    d_beta = np.zeros(beta.shape)
    num_data = len(data["user_id"])
    for index in range(num_data):
        answer = data["is_correct"][index]
        j = data["question_id"][index]
        i = data["user_id"][index]
        d = theta[i] - beta[j]
        if answer == 1:
            d_theta[i] += 1 / (1+np.exp(d))
            d_beta[j] += -1 / (1+np.exp(d))
        elif answer == 0:
            d_theta[i] += 1 / (1+np.exp(d)) - 1
            d_beta[j] += -1 / (1+np.exp(d)) + 1
    # Here we want to maximize likelihood which is a concave function, instead
    # of decent, we actually accent.
    theta += lr*d_theta
    beta += lr*d_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.zeros(1+max(data["user_id"]))
    beta = np.zeros(1+max(data["question_id"]))

    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

        train_lld_lst.append(neg_lld)
        val_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_lld_lst.append(val_lld)

    return theta, beta, val_acc_lst, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 20

    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = \
        irt(train_data, val_data, lr, iterations)
    iterations_lst = list(range(iterations))

    plt.plot(iterations_lst, train_lld_lst, "r-", label="training")
    plt.plot(iterations_lst, val_lld_lst, "b-", label="validation")
    plt.xlabel("Iterations")
    plt.ylabel("Negative likelihood")
    plt.legend()
    plt.show()

    plt.plot(iterations_lst, val_acc_lst)
    plt.xlabel("Iterations")
    plt.ylabel("Validation accuracy")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Implement part (c)
    #####################################################################
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Validation accuracy: {}".format(val_acc))
    print("Test accuracy: {}".format(test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
