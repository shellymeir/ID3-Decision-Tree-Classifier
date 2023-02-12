from sklearn.model_selection import KFold
import numpy as np
import myUtils


class PersonalizedID3:

    def __init__(self):
        self.train_set = None
        self.decision_tree_root = None

    def learn(self, train_set, M=1, ratio=0.9):
        self.train_set = train_set
        self.decision_tree_root = myUtils.Vertex(self.train_set)
        myUtils.developDecisionTree(self.decision_tree_root, 1, is_personalized=True, personalized_ratio=ratio)

    def classify(self, data, threshold=50):
        return myUtils.classifyAux(self.decision_tree_root, data, is_personalized=True, threshold=threshold)

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
