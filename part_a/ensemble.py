import part_a.item_response as m2ir

from utils import *


def train_model_ir(train_data):
    print("-------------training IRT start-----------")
    val_data = load_valid_csv("../data")
    lr = 0.01
    iterations = 20

    theta, beta, val_acc_lst, train_lld_lst, val_lld_lst = \
        m2ir.irt(train_data, val_data, lr, iterations)
    print("-------------training IRT complete-----------")
    return theta, beta


def predict_ir(data, theta, beta):
    prob = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = m2ir.sigmoid(x)
        prob.append(p_a)
    return prob


def f(lst):
    # return lst
    t = []
    for i in np.arange(len(lst)):
        t.append(1) if lst[i] >= 0.5 else t.append(0)
    return t


def sample(ratio=1):
    data = load_train_csv("../data")
    size = len(data["question_id"])
    resample = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    for i in np.random.choice(size, int(np.floor(size * ratio))):
        resample["user_id"].append(data["user_id"][i])
        resample["question_id"].append(data["question_id"][i])
        resample["is_correct"].append(data["is_correct"][i])
    return resample


def main():
    theta1, beta1 = train_model_ir(sample())
    theta2, beta2 = train_model_ir(sample())
    theta3, beta3 = train_model_ir(sample())

    # model, d = train_model_nn(sample())
    # mat = train_model_knn(sample())

    def evaluation(data):
        p_irt1 = predict_ir(data, theta1, beta1)
        p_irt2 = predict_ir(data, theta2, beta2)
        p_irt3 = predict_ir(data, theta3, beta3)

        p_irt1 = f(p_irt1)
        p_irt2 = f(p_irt2)
        p_irt3 = f(p_irt3)

        pred = []
        for i in np.arange(len(p_irt1)):
            temp = (p_irt1[i] + p_irt2[i] + p_irt3[i]) / 3
            pred.append(temp >= 0.5)
        acc = np.sum((data["is_correct"] == np.array(pred))) \
            / len(data["is_correct"])
        print(acc)

    evaluation(load_valid_csv("../data"))
    evaluation(load_public_test_csv("../data"))


if __name__ == "__main__":
    main()
