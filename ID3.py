from sklearn.model_selection import KFold
import numpy as np
import myUtils
import matplotlib.pyplot as plt


class ID3:

    def __init__(self):
        self.train_set = None
        self.decision_tree_root = None

    def learn(self, train_set, M=1):
        self.train_set = train_set # [[x for x in row] for row in train_set]
        self.decision_tree_root = myUtils.Vertex(self.train_set)
        myUtils.developDecisionTree(self.decision_tree_root, M)

    def classify(self, data):
        return myUtils.classifyAux(self.decision_tree_root, data)

    def fit_predict(self, train, test) -> np.ndarray:
        train_set = myUtils.npArrayToListOfLists(train, 1)
        test_set = myUtils.npArrayToListOfLists(test, 0)
        self.learn(train_set)  # create decision tree

        result = []
        for person in range(len(test_set)):
            if self.classify(test_set[person]) == 'M':
                result.append(1)  # person is ill
            else:
                result.append(0)  # person is healthy

        result = np.array(result)
        return result

    def estimatePerformanceKFold(self, train_set, M=1):
        train_set = myUtils.npArrayToListOfLists(train_set, 1)
        kf = KFold(n_splits=5, shuffle=True, random_state=205905490)
        results = []
        for train_index, test_index in kf.split(train_set):
            train_group = [[x for x in train_set[i]] for i in train_index]
            test_group = {}
            for i in test_index:
                test_group[i] = [x for x in train_set[i]]
            test_res = {}
            for person in test_group.values():
                del person[0]  # remove mark from test group
            self.learn(train_group, M)
            for person in test_group.keys():
                test_res[person] = self.classify(test_group[person])

            results.append(test_res)

        return results

    def calcLoss(self, train_array, M=1, false_negative_weight=1, false_positive_weight=1):
        test_results = self.estimatePerformanceKFold(train_array, M)
        loss = []
        for test_result in test_results:
            curr_loss = 0
            for person in test_result.keys():
                if test_result[person] != train_array[person][0]:
                    if test_result[person] == 'B':  # False Negative
                        curr_loss += false_negative_weight
                    else:
                        curr_loss += false_positive_weight
            loss.append(curr_loss)

        return sum(loss) / len(loss)


def experiment(id3_obj, train_set):
    """
    :param: id3_obj : ID3 object
    :param: train_set : train set of type np.array, like the one we use in fit_predict function
    the last command in this function is marked as a comment as asked in the pdf, uncomment it to view the created graph
    """
    M_values = [i for i in range(1, 55, 5)]

    results = {}
    for M in M_values:
        results[M] = myUtils.calcSuccessPercentage(train_set, id3_obj.estimatePerformanceKFold(train_set, M))

    x = [M for M in M_values]
    y = [success_rate for success_rate in results.values()]
    plt.plot(x, y, color='black', linewidth=0.3)
    plt.scatter(x, y, color='magenta', marker='*', s=15)
    plt.xlabel('M value')
    plt.ylabel('Success Percentage')
    plt.title('Early Pruning: Success Percentage as a function of M Value')
    # plt.show()

