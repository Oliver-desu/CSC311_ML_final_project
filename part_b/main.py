from part_a.item_response import *
from part_b.utils_b import classify_subjects, classify_question


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
    """ Train IRT model by data from different subject.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta: dim num_student X num_subject, beta: dim num_question
    X num_subject)
    """
    subjects = classify_subjects(data)
    num_subject = len(subjects)
    theta = np.full((1+max(data["user_id"]), num_subject), np.nan)
    beta = np.full((1+max(data["question_id"]), num_subject), np.nan)

    for i in range(num_subject):
        subject_data = subjects[i]
        theta_sub, beta_sub = simple_irt(subjects[i], lr, iterations)
        for q in subject_data["question_id"]:
            beta[q, i] = beta_sub[q]
        for u in subject_data["user_id"]:
            theta[u, i] = theta_sub[u]
    return theta, beta


def _test():
    data1 = load_train_csv("../data")
    theta, beta = subject_irt(data1, 0.01, 20)
    print(beta[0])


if __name__ == "__main__":
    _test()
