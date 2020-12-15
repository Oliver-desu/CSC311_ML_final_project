import torch

import part_a.knn as m1knn
import part_a.item_response as m2ir
import part_a.neural_network as m3nn

from utils import *
from sklearn.impute import KNNImputer
from torch.autograd import Variable


def train_model_ir(train_data):
    print("-------------training IRT start-----------")
    val_data = load_valid_csv("../data")
    lr = 0.01
    iterations = 20

    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = \
        m2ir.irt(train_data, val_data, lr, iterations)
    print("-------------training IRT complete-----------")
    return theta, beta


def train_model_nn(train_data):
    print("-------------training NN start-----------")
    train_data = load_train_csv("../data")
    train_matrix = load_train_sparse("../data").toarray()
    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)

    zero_train_matrix, train_matrix, valid_data, test_data = m3nn.load_data()
    num_questions = train_matrix.size()[1]
    k_set = [10, 50, 100, 200, 500]
    lamb_set = [0, 0.001, 0.01, 0.1, 1]
    # Set model hyperparameters.
    k = k_set[1]
    model = m3nn.AutoEncoder(num_questions, k)

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 43
    lamb = lamb_set[3]

    m3nn.train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
    print("-------------training NN complete-------------")
    return model, zero_train_matrix


def train_model_knn(train_data):
    print("-------------training KNN start-----------")
    matrix = load_train_sparse("../data").toarray()
    k = 11
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    print("-------------training KNN complete-----------")
    return mat


def predict_knn(data, mat):
    prob = []
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        prob.append(mat[cur_user_id, cur_question_id])
    return prob


def predict_ir(data, theta, beta):
    prob = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = m2ir.sigmoid(x)
        prob.append(p_a)
    return prob


def predict_nn(model, train_data, valid_data):
    prob = []
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][valid_data["question_id"][i]].item()
        prob.append(guess)
    return prob


def f(lst):
    # return lst
    t = []
    for i in np.arange(len(lst)):
        t.append(1) if lst[i] >= 0.5 else t.append(0)
    return t


def sample():
    data = load_train_csv("../data")
    size = len(data["question_id"])
    resample = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    for i in np.random.choice(size, int(np.floor(size / 3))):
        resample["user_id"].append(data["user_id"][i])
        resample["question_id"].append(data["question_id"][i])
        resample["is_correct"].append(data["is_correct"][i])
    return resample


def main():

    theta, beta = train_model_ir(sample())
    model, d = train_model_nn(sample())
    mat = train_model_knn(sample())

    def evaluation(data):
        p_nn = predict_nn(model, d, data)
        p_irt = predict_ir(data, theta, beta)
        p_knn = predict_knn(data, mat)
        p_nn = f(p_nn)
        p_irt = f(p_irt)
        p_knn = f(p_knn)
        pred = []
        for i in np.arange(len(p_nn)):
            temp = p_nn[i] + p_knn[i] + p_irt[i] / 3
            pred.append(temp >= 0.5)
        acc = np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])
        print(acc)
    evaluation(load_valid_csv("../data"))
    evaluation(load_public_test_csv("../data"))


if __name__ == "__main__":
    main()
