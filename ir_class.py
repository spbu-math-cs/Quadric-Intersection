import numpy as np
import itertools
from tqdm import tqdm


def calculate_inv_dict(labels):
    """Returns dictionary {label: array of position}

    Examples
    >>> calculate_inv_dict([1, 1, 2, 3, 3])
    {1:[0, 1], 2:[2], 3:[3, 4]}
    """

    inv_dict = {}
    for index, i in enumerate(labels):
        if i not in inv_dict.keys():
            inv_dict[i] = [index]
        else:
            inv_dict[i].append(index)
    return inv_dict


def search_for_fp_examples(arr_sorted, n_threshold):
    l = np.where(np.cumsum(1 - arr_sorted[::-1]) == n_threshold)[0][0]
    return l - n_threshold + 1


def manual_roc_score(arr, ds_issame, *fpr_arr):
    s = np.argsort(arr)
    sorted_issame = ds_issame[s]
    sorted_arr = arr[s]
    answer = []

    for f in fpr_arr:
        FP = int(np.sum(ds_issame == 0) * f)
        TP = search_for_fp_examples(sorted_issame, FP)
        answer.append(sorted_arr[-FP - TP])
    return answer


def similarity_transformed(data1, data2, feature1, feature2, alpha,
                           dist_type="dot_product"):
    """Consider data normed to 1. Features have the shape of len(data).
       Returns similarity array of shape (len(data1), len(data2))"""
    transformation_types = {"dot_product", "max", "min", "fraction",
                            "multiplication", "sum", "sum_threshold",
                            "max_threshold", "min_threshold",
                            "max_threshold_below", "min_threshold_below"}
    assert alpha in transformation_types
    feature1_full = np.repeat(feature1, len(feature2)).reshape(
        (len(feature1), len(feature2)))
    if dist_type == "dot_product":
        similarity = np.dot(data1, data2.T) - alpha * np.outer(feature1,
                                                               feature2)
    if dist_type == "max":
        similarity = np.dot(data1, data2.T) - alpha * np.maximum(feature1_full,
                                                                 feature2)
    if dist_type == "min":
        similarity = np.dot(data1, data2.T) - alpha * np.minimum(feature1_full,
                                                                 feature2)
    if dist_type == "fraction":
        similarity = np.dot(data1, data2.T) / (
                    1 + alpha * np.minimum(feature1_full, feature2))
    if dist_type == "multiplication":
        similarity = np.dot(data1, data2.T) * (
                    1 - alpha * np.minimum(feature1_full, feature2))
    if dist_type == "sum":
        similarity = np.dot(data1, data2.T) - alpha * (
                    feature1_full - feature2)
    if dist_type == "sum_threshold":
        feature1_full = np.subtract.outer(feature1, -feature2) < alpha
        similarity = np.dot(data1, data2.T) * feature1_full
    if dist_type == "max_threshold":
        feature1_full = np.maximum(feature1_full, feature2) < alpha
        similarity = np.dot(data1, data2.T) * feature1_full
    if dist_type == "min_threshold":
        feature1_full = np.minimum(feature1_full, feature2) < alpha
        similarity = np.dot(data1, data2.T) * feature1_full
    if dist_type == "max_threshold_below":
        feature1_full = np.maximum(feature1_full, feature2) > alpha
        similarity = np.dot(data1, data2.T) * feature1_full
    if dist_type == "min_threshold_below":
        feature1_full = np.minimum(feature1_full, feature2) > alpha
        similarity = np.dot(data1, data2.T) * feature1_full
    return similarity


class IrBigData:
    """
    Class for identification rate calculation.


    Parameters
    ----------
    data : ndarray, embeddings of shape (n_data_samples, n_data_features)

    data_features : ndarray, features for embeddings of shape (n_data_features, )

    labels : ndarray, labels for each embedding of shape (n_data_features, )

    distractors : ndarray of distractor embeddings
                  of shape (n_distractors_samples, n_distractors_features)

    distractor_features : ndarray of distractors features
                          of shape (n_distractors_features, )

    parameters : dict of paramters:
        similarity_type : str, can be one of two values: dot_product, features

        fpr_threshold : float, false positive rate for IR calculation

        dist_type : str, type of similarity transformation,
                    for details check function similarity_transformed

        protocol : str, either data_distractors_in_false_pairs,
                   data_distractors_no_false_pairs, data_no_distractors

    Attributes
    ----------
    s_f_ : float, threshold in distances corresponding to given fpr_threshold

    pairs_true_ : list of tuples of true pairs

    pairs_false_threshold_ : list, false negative pairs below s_f_

    pairs_false_distractors_ : list, false negative pairs distracted by the
                               set of distractors

    CMT_ : float, calculated identification rate

    """

    def __init__(self, data, data_features, labels, parameters,
                 distractors=None, distractor_features=None):
        self.data = data
        self.data_features = data_features
        self.labels = labels
        self.params = parameters
        self.distractors = distractors
        self.distractor_features = distractor_features
        self.check_dimension()

    def get_distances_transformed(self, dist_type="dot_product",
                                  alpha=None):
        if not alpha:
            alpha = self.params["alpha"]
        if self.params["similarity_type"] == "dot_product":
            # distances here are scalar products
            self._distances = np.dot(self.data, self.data.T)
            if isinstance(self.distractors, np.ndarray):
                self._distances_distractors = np.dot(self.data,
                                                     self.distractors.T)

        if self.params["similarity_type"] == "features":
            self._distances = similarity_transformed(self.data, self.data,
                                                     self.data_features,
                                                     self.data_features, alpha,
                                                     dist_type=self.params["dist_type"])
            if isinstance(self.distractors, np.ndarray):
                self._distances_distractors = similarity_transformed(self.data,
                                                                     self.distractors,
                                                                     self.data_features,
                                                                     self.distractor_features,
                                                                     alpha,
                                                                     dist_type=self.params["dist_type"])

    def prepare_info_arrays(self):
        self.errors_distractors_dict_ = dict()
        self.errors_threshold_set_ = set()
        self.pairs_false_threshold_ = set()  # pairs with inner distanse > threshold
        self.pairs_false_distractors_ = set()  # pairs with with inner distanse > distance to some distractor

    def final_func(self):
        self.inv_dict_ = calculate_inv_dict(self.labels)
        # get distances (similarity) between points
        self.get_distances_transformed(dist_type=self.params["dist_type"],
                                       alpha=self.params["alpha"])
        self.prepare_pairs()
        # manually calculate threshold s_f_
        self.s_f_ = manual_roc_score(
            np.hstack((self._distances_true, self._distances_false)),
            self._ds_issame, self.params["fpr_threshold"])[0]
        self.ir_with_distractors()
        self.print_info()

    def prepare_pairs(self):
        self.pairs_true_ = []
        # compose true pairs
        for label in self.inv_dict_:
            self.pairs_true_.extend(
                list(itertools.permutations(self.inv_dict_[label], r=2)))

        # get distances for true and false pairs
        self._distances_true = np.array(
            [self._distances[pair] for pair in self.pairs_true_])
        distances_extra = np.copy(self._distances)
        for i in self.pairs_true_:
            distances_extra[i] = 0
        np.fill_diagonal(distances_extra, 0)

        distances_extra = distances_extra.flatten()
        self._distances_false = distances_extra[distances_extra != 0]

        # add distractors in flase pairs
        if self.params['protocol'] == 'data_distractors_in_false_pairs':
            self._distances_false = np.hstack(
                (self._distances_false, self._distances_distractors.flatten()))

        self._ds_issame = np.hstack(
            ([1 for i in range(len(self._distances_true))],
             [0 for i in range(len(self._distances_false))]))

    def check_dimension(self):
        assert len(self.data) == len(self.data_features) == len(self.labels)
        assert isinstance(self.data, np.ndarray)
        if self.params["protocol"] == "no_distractors":
            assert self.params["similarity_type"] == "dot_product"
        if isinstance(self.distractors, np.ndarray):
            assert len(self.distractors) == len(self.distractor_features)

    def ir_with_distractors(self, s_f=None, extra_info=True):
        CMT = 0
        self.disturbed_distractors_ = []
        if extra_info:
            self.prepare_info_arrays()
        if not s_f:
            s_f_ = self.s_f_

        for index, pair in tqdm(enumerate(self.pairs_true_)):
            if self.params["protocol"] == "data_no_distractors":
                condition_1 = True
            else:
                condition_1 = np.all(self._distances[pair] > self._distances_distractors[pair[0]])
                self.disturbed_distractors_.append(np.nonzero(self._distances[pair] <
                                                   self._distances_distractors[pair[0]])[0])
            if self._distances[pair] > s_f_ and condition_1:
                CMT += 1
            if extra_info:
                # check false pairs
                if self._distances[pair] < s_f_:
                    self.pairs_false_threshold_.add(index)
                    self.errors_threshold_set_.add(pair)
                if not condition_1:
                    self.pairs_false_distractors_.add(index)
                    self.errors_distractors_dict_[pair] = self._distances[
                                                              pair] < \
                                                          self._distances_distractors[
                                                              pair[0]]
        CMT /= len(self.pairs_true_)
        self.CMT_ = CMT

    def print_info(self):
        print("id rate =", self.CMT_)
        print("id rate no distractors =",
              (len(self.pairs_true_) - len(self.pairs_false_threshold_)) / len(
                  self.pairs_true_))
        print("id rate only distractors =",
              (len(self.pairs_true_) - len(
                  self.pairs_false_distractors_)) / len(self.pairs_true_))


