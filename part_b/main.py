from part_a.item_response import *
from part_b.utils_b import classify_subjects, classify_question, load_subject


def simple_irt(data, lr, iterations):
    """ Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta)
    """
    if len(data["user_id"]) == 0:
        return np.zeros(0), np.zeros(0)
    theta = np.zeros(1+max(data["user_id"]))
    beta = np.zeros(1+max(data["question_id"]))

    for i in range(iterations):
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta


def subject_irt(data, lr, iterations):
    """ Train IRT model by data from different subject, last column is overall
    IRT without split by subjects.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta: dim num_student X (num_subject+1), beta: dim num_question
    X (num_subject+1))
    """
    subjects = classify_subjects(data)
    num_subject = len(subjects)
    theta = np.full((1+max(data["user_id"]), num_subject+1), np.nan)
    beta = np.full((1+max(data["question_id"]), num_subject+1), np.nan)

    for i in range(num_subject):
        subject_data = subjects[i]
        theta_sub, beta_sub = simple_irt(subjects[i], lr, iterations)
        for q in subject_data["question_id"]:
            beta[q, i] = beta_sub[q]
        for u in subject_data["user_id"]:
            theta[u, i] = theta_sub[u]
    theta[:, -1], beta[:, -1] = simple_irt(data, lr, iterations)
    return theta, beta


def probability(diff, weights):
    """ return probability of some student i solving some question j correctly

    :param diff: ability of student i minus difficulty of question j for
    different subject
    :param weights: how much knowledge question j depend on different subject
    :return: probability of student i giving correct answer to question j
    """
    d = 0
    for i in diff:
        d += diff[i] * weights[i]
    return sigmoid(d)


def _helper_grad(diff, weights, correct):
    # helper function of grad
    left = correct - probability(diff, weights)
    result = {}
    for i in diff:
        result[i] = -left*diff[i]
    return result


def dict_add(dict1, dict2):
    for i in dict1:
        dict1[i] += dict2[i]


def normalize(weights):
    f = 0
    for i in weights:
        f += weights[i]
    for i in weights:
        weights[i] /= f


class AdvanceIRT:
    def __init__(self, data, subject_dict, theta, beta):
        self.data = data
        self.subject_dict = subject_dict
        self.theta = theta
        self.beta = beta
        self.q_weights = {}

    def validation(self, val_data):
        # return validation accuracy
        count = 0
        total = len(val_data["question_id"])
        for i, q in enumerate(val_data["question_id"]):
            s = val_data["user_id"][i]
            prediction = self.predict(q, s, self.subject_dict[q])
            if prediction == val_data["is_correct"][i]:
                count += 1
        return count/total

    def compare(self, val_data):
        for question_id in self.subject_dict:
            data = classify_question(val_data, question_id)
            if len(data["user_id"]) == 0:
                continue
            weights = self.q_weights[question_id]
            subjects = self.subject_dict[question_id]
            self.q_weights[question_id] = {-1: 1}
            for i in subjects:
                self.q_weights[question_id][i] = 1
            acc1 = self.validation(data)
            self.q_weights[question_id] = weights
            acc2 = self.validation(data)
            print(question_id, acc2 - acc1)

    def predict(self, question_id, student_id, subjects):
        # predict whether student i can solve question j correctly
        weights = self.q_weights[question_id]
        diff = self.ability_difference(question_id, student_id, subjects)
        p = probability(diff, weights)
        if p < 0.5:
            return 0
        else:
            return 1

    def train3(self, lr, iterations, regulation):
        """ Train the model by gradient decent

        :param lr: learning rate
        :param iterations: number of iterations
        :param regulation: regulation coefficient
        """
        for q in self.subject_dict:
            self.gradient_decent(q, lr, iterations, regulation)

    def gradient_decent(self, question_id, lr, iterations, regulation):
        # perform gradient decent on some question to determine dependence of
        # each subject and store it in self.q_weights

        data = classify_question(self.data, question_id)
        subjects = self.subject_dict[question_id]
        weights = {-1: 1}
        for i in subjects:
            weights[i] = 1

        for iteration in range(iterations):
            grad = self.grad(question_id, data["user_id"], data["is_correct"],
                             subjects, weights, regulation)
            for i in grad:
                weights[i] -= lr*grad[i]
                if weights[i] < 0.01:
                    weights[i] = 0.01
                elif weights[i] > 20:
                    weights[i] = 20
        self.q_weights[question_id] = weights
        # print(question_id, weights)

    def grad(self, question_id, student_lst, correct_lst, subjects,
             initial_weights, regulation):
        # compute gradient
        grad = {-1: regulation*initial_weights[-1]}
        for i in subjects:
            grad[i] = regulation*initial_weights[i]

        for index, s in enumerate(student_lst):
            diff = self.ability_difference(question_id, s, subjects)
            dict_add(grad, _helper_grad(diff, initial_weights,
                                        correct_lst[index]))
        return grad

    def ability_difference(self, question_id, student_id, subjects):
        # return ability of student minus difficulty of question in each subject

        diff = {-1: self.theta[student_id, -1] - self.beta[question_id, -1]}
        for i in subjects:
            if np.isnan(self.theta[student_id, i]):
                continue
            diff[i] = self.theta[student_id, i] - self.beta[question_id, i]
        return diff


def save(file_name, data):
    """ Saves the model to a numpy file.
    """
    print("Writing to " + file_name)
    np.savez_compressed(file_name, data)


def load(file_name):
    """ Loads the model from numpy file.
    """
    print("Loading from " + file_name)
    return dict(np.load(file_name))


def _test():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    subject_dict = load_subject()

    # theta, beta = subject_irt(train_data, 0.01, 20)
    # save("theta", theta)
    # save("beta", beta)
    theta, beta = load("theta.npz")["arr_0"], load("beta.npz")["arr_0"]

    model = AdvanceIRT(train_data, subject_dict, theta, beta)
    model.train3(1, 50, 1)
    model.compare(val_data)
    acc = model.validation(val_data)
    print(acc)

    # # Observe ability difference for question 0
    # data2 = classify_question(train_data, 0)
    # for i, s in enumerate(data2["user_id"]):
    #     lst = [data2["is_correct"][i]]
    #     for j in load_subject()[0]:
    #         lst.append(model.theta[s, j]-model.beta[0, j])
    #     print(lst)


if __name__ == "__main__":
    _test()
