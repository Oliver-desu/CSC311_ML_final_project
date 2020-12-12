from utils import *


def _load_subject(path):
    # Helper function to load subject
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data[int(row[0])] = str_to_lst(row[1])
            except ValueError:
                # Pass first row.
                pass
    return data


def str_to_lst(s):
    """ Convert str to list. For example:
    >>> s = '[0, 1, 5, 98, 147, 167]'
    >>> str_to_lst(s)
    >>> [0, 1, 5, 98, 147, 167]
    """
    return [int(i) for i in s.strip("[ ]").split(", ")]


def load_subject(root_dir="../data"):
    """ Load the subject information of questions.
    It's in form of a dictionary where:
    key is question id, value is list of subject id
    """
    path = os.path.join(root_dir, "question_meta.csv")
    return _load_subject(path)


def classify_subjects(data):
    """ Evaluate the model given data and return the accuracy.
        :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
        :return: A dictionary {int: data}
        """
    # According to subject_meta, subjects id through 0 - 387.
    num_subject = 388
    subject_data = {}
    for i in range(num_subject):
        subject_data[i] = {"user_id": [], "question_id": [], "is_correct": []}

    subject_dict = load_subject()
    for i, q in enumerate(data["question_id"]):
        for subject in subject_dict[q]:
            subject_data[subject]["user_id"].append(data["user_id"][i])
            subject_data[subject]["question_id"].append(q)
            subject_data[subject]["is_correct"].append(data["is_correct"][i])
    return subject_data


def _test():
    data1 = load_train_csv("../data")
    print(classify_subjects(data1)[0])


if __name__ == "__main__":
    _test()