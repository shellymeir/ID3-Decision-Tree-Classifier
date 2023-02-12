from csv import reader
from math import log2
import numpy as np


def readCsvFile(filename):
    """
    :return: list of lists of the file 'filename.csv'
    """
    with open(filename + '.csv', 'r') as file:
        content = reader(file)
        res = list(content)
        file.close()

    res = np.array(res)
    return res


def npArrayToListOfLists(array, start_index):
    res = array.tolist()
    for i in range(len(res)):
        for j in range(start_index, len(res[0])):
            res[i][j] = float(res[i][j])

    return res


def featureMaxIG(marked_feature_examples):
    """
    for continuous features: sort all k examples, compute k-1 thresholds ('new features'),
    calculate IG for each threshold
    :return: a tuple of threshold-feature with max IG and its IG
    """
    num_of_examples = len(marked_feature_examples)
    if num_of_examples == 0:
        return 0

    # create a sorted list of marked examples of the given feature
    sorted_examples = marked_feature_examples
    sorted_examples = sorted(sorted_examples, key=lambda x: x[1])  # sort examples according to their feature value
    # create new k-1 discrete features (where k is number of examples)
    thresholds = []
    for i in range(len(sorted_examples)-1):
        thresholds.append((sorted_examples[i][1]+sorted_examples[i+1][1])/2)

    # calculate the new sets we get by dividing the examples with each feature
    new_sets = []  # (<threshold, >=threshold)
    less_than_threshold = []
    greater_or_equal_to_threshold = []
    for threshold in thresholds:
        for example in sorted_examples:  # example is a tuple of (class, feature value)
            if example[1] < threshold:
                less_than_threshold.append(example)
            else:
                greater_or_equal_to_threshold.append(example)
        new_sets.append((less_than_threshold, greater_or_equal_to_threshold))
        less_than_threshold = []
        greater_or_equal_to_threshold = []

    # calculate IG for each new feature
    ent = entropy(sorted_examples)  # calculate the Entropy before the division
    new_features_IG = {}
    for threshold in range(len(thresholds)):
        lt_size = len(new_sets[threshold][0])  # num of examples which are less than threshold
        ge_size = len(new_sets[threshold][1])  # num of examples which are greater than or equal to threshold
        lt_ent = entropy(new_sets[threshold][0])
        ge_ent = entropy(new_sets[threshold][1])
        curr_threshold_IG = ent - ((lt_size/num_of_examples)*lt_ent + (ge_size/num_of_examples)*ge_ent)
        new_features_IG[thresholds[threshold]] = curr_threshold_IG

    # find the new feature with max IG
    max_feature_IG = (0, 0)
    for item in new_features_IG.items():
        if item[1] >= max_feature_IG[1]:
            max_feature_IG = item

    return max_feature_IG


def entropy(examples):
    """
    calculate entropy of a given set
    :param examples: a set of example-tuples where the first element is the example's class and the second is its f-val
    :return: the entropy of the set
    """
    if len(examples) == 0:
        return 0

    sick_count = 0
    healthy_count = 0
    for person_details in examples:
        if person_details[0] == 'M':  # person is classified as sick
            sick_count += 1
        else:
            healthy_count += 1

    sick_prob = sick_count / len(examples)
    healthy_prob = healthy_count / len(examples)

    sick_res = 0
    if sick_prob > 0:
        sick_res = sick_prob * log2(sick_prob)

    healthy_res = 0
    if healthy_prob > 0:
        healthy_res = healthy_prob * log2(healthy_prob)
    return -(healthy_res + sick_res)


class Vertex:
    def __init__(self, objects):
        self.objects = [[x for x in row] for row in objects]  # objects are given as a list of lists where each list describes all information about
                                # a person such that the first value in list is the person's class: sick or healthy, and
                                # the rest values are the different features values
        self.successors = []
        self.feature = None
        self.feature_threshold = None
        self.objects_class = None  # for leafs

    def selectFeature(self):
        if len(self.objects) == 0:
            return None

        # prepare marked examples for each feature
        features_info = {}  # dict of feature:list where list contains tuples of (class, feature val) for each person
        feature_list = []
        for feature in range(1, len(self.objects[0])):
            for person in self.objects:
                feature_list.append((person[0], person[feature]))
            features_info[feature] = feature_list
            feature_list = []

        # find feature with max IG, if more than 1 with same value, return the latter
        max_feature_IG = (0, 0, 0)  # (feature, threshold, IG)
        for feature in features_info.keys():
            feature_threshold_IG = featureMaxIG(features_info[feature])
            if feature_threshold_IG[1] >= max_feature_IG[2]:
                max_feature_IG = (feature, feature_threshold_IG[0], feature_threshold_IG[1])
        return max_feature_IG[0], max_feature_IG[1]  # return tuple (chosen feature, threshold to divide its vals with)

    def isHomogeneous(self, is_personalized=False, ratio=None):
        if is_personalized:
            M_count = 0
            B_count = 0
            for person in self.objects:
                if person[0] == 'M':
                    M_count += 1
                else:
                    B_count += 1

            if M_count >= ratio*(M_count+B_count) or B_count == 0:
                self.objects_class = 'M'
                return True
            elif M_count == 0:
                self.objects_class = 'B'
                return True
            return False
        vertex_class = self.objects[0][0]
        for person in self.objects:
            if person[0] != vertex_class:
                return False
        self.objects_class = vertex_class
        return True

    def getDividedGroups(self):
        lt_threshold = []
        ge_threshold = []
        for person in self.objects:
            if person[self.feature] < self.feature_threshold:
                lt_threshold.append(person)
            else:
                ge_threshold.append(person)

        return lt_threshold, ge_threshold

    def getMajorityClass(self):
        M_count = 0
        B_count = 0
        for person in self.objects:
            if person[0] == 'M':
                M_count += 1
            else:
                B_count += 1

        if M_count > B_count:
            return 'M'
        else:
            return 'B'


def developDecisionTree(vertex, M=1, is_personalized=False, personalized_ratio=None):

    if vertex.isHomogeneous(is_personalized, personalized_ratio):  # if True vertex class gets its value inside this func
        return

    if len(vertex.objects) < M:  # for pruning
        vertex.objects_class = vertex.getMajorityClass(is_personalized)  # vertex is a leaf
        return

    divider_feature = vertex.selectFeature()
    vertex.feature = divider_feature[0]
    vertex.feature_threshold = divider_feature[1]
    successors = vertex.getDividedGroups()

    vertex.successors.append(Vertex(successors[0]))
    vertex.successors.append(Vertex(successors[1]))

    for successor in vertex.successors:
        developDecisionTree(successor, M, is_personalized, personalized_ratio)


def enlargeSickInGroupFactor(groups):
    lt_sick_num = 0
    ge_sick_num = 0
    for person in groups[0]:
        if person[0] == 'M':
            lt_sick_num += 1
    for person in groups[1]:
        if person[0] == 'M':
            ge_sick_num += 1
    if lt_sick_num > ge_sick_num:
        return 1.01
    else:
        return 0.99


def postPrune(tree_root, validation_group):
    if len(tree_root.successors) == 0:
        return tree_root
    left_validation_group = [person for person in validation_group if person[tree_root.feature] < tree_root.feature_threshold]
    right_validation_group = [person for person in validation_group if person[tree_root.feature] >= tree_root.feature_threshold]
    tree_root.successors[0] = postPrune(tree_root.successors[0], left_validation_group)
    tree_root.successors[1] = postPrune(tree_root.successors[1], right_validation_group)

    error_prune = 0
    error_no_prune = 0
    for person in validation_group:
        error_prune += evaluate(person[0], tree_root.getMajorityClass())
        error_no_prune += evaluate(person[0], classifyAux(tree_root, [person[i] for i in range(1, len(person))]))
    if error_prune <= error_no_prune:
        tree_root.feature = None
        tree_root.feature_threshold = None
        tree_root.objects_class = tree_root.getMajorityClass()
        tree_root.successors = []

    return tree_root


def evaluate(v_class, t_class):
    if v_class != t_class:
        return 8 if v_class == 'M' else 1
    else:
        return 0


def classifyAux(vertex, data, is_personalized=False, threshold=50):
    if vertex.objects_class is not None:
        return vertex.objects_class  # leaf
    if is_personalized and len(vertex.objects) < threshold:
        return nearest_neighbor(vertex.objects, data)
    if data[vertex.feature-1] < vertex.feature_threshold:
        return classifyAux(vertex.successors[0], data, is_personalized, threshold)
    else:
        return classifyAux(vertex.successors[1], data, is_personalized, threshold)


def calcSuccessPercentage(examples, test_results):
    success_percentages = []
    for res in test_results:
        cntr = 0
        for person in res.keys():
            if res[person] == examples[person][0]:
                cntr += 1
        success_percentages.append(cntr / len(res))
    return sum(success_percentages) / len(success_percentages)


def calcLoss(test_results, real_results):
    counter = 0
    for i in range(test_results.size):
        if test_results[i] == 1 and real_results[i][0] == 'B':
            counter += 1
        elif test_results[i] == 0 and real_results[i][0] == 'M':
            counter += 8
    return counter


def nearest_neighbor(list_of_lists, checked_list):
    # assumption list_of_lists with 0 classification , checked_list without the classification
    max_of_every_feature = {}
    min_of_every_feature = {}
    normalized_list_of_lists = []
    diffs_list_of_samples = {}
    for sample in range(0, len(list_of_lists)):
        for feature in range(1, len(list_of_lists[0])):
            if feature not in max_of_every_feature:  # defining max
                max_of_every_feature[feature] = list_of_lists[sample][feature]
            else:
                if max_of_every_feature[feature] < list_of_lists[sample][feature]:
                    max_of_every_feature[feature] = list_of_lists[sample][feature]
            if feature not in min_of_every_feature:  # defining min
                min_of_every_feature[feature] = list_of_lists[sample][feature]
            else:
                if min_of_every_feature[feature] > list_of_lists[sample][feature]:
                    min_of_every_feature[feature] = list_of_lists[sample][feature]

    for sample in range(0, len(list_of_lists)):
        new_list = []
        for feature in range(1, len(list_of_lists[0])):
            added_value = (list_of_lists[sample][feature] - min_of_every_feature[feature]) / (
                        max_of_every_feature[feature] - min_of_every_feature[feature] + 0.01)
            new_list.append(added_value)
        normalized_list_of_lists.append(
            new_list)  # now the Normalized_list_of_lists is starting from 0 (and normalized)
    i = 0
    for normalized_sample in normalized_list_of_lists:
        diffs_list_of_samples[i] = 0
        for feature in range(0, len(normalized_sample)):
            normalized_element_in_check_list = (checked_list[feature] - min_of_every_feature[feature + 1]) / (
                        max_of_every_feature[feature + 1] - min_of_every_feature[feature + 1] + 0.01)
            diffs_list_of_samples[i] += abs(normalized_element_in_check_list - normalized_sample[feature])
        i += 1
    min_of_diffs_list_of_samples_index = 0
    min_of_diffs_list_of_samples_value = float('inf')
    for i in diffs_list_of_samples.keys():
        if min_of_diffs_list_of_samples_value > diffs_list_of_samples[i]:
            min_of_diffs_list_of_samples_value = diffs_list_of_samples[i]
            min_of_diffs_list_of_samples_index = i
    return list_of_lists[min_of_diffs_list_of_samples_index][0]


