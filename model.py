__author__ = 'alfiya'

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
# from skll.metrics import kappa
import logging
from six import string_types
from sklearn.metrics import make_scorer, confusion_matrix
import enchant
from nltk.stem.porter import PorterStemmer
from difflib import SequenceMatcher
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cross_validation import train_test_split
import time
import subprocess
import csv
from sklearn.cluster import MeanShift
from sklearn.pipeline import Pipeline


def time_passed(start):
    return round((time.time() - start) / 60)

def numpy2libfm(X, y=None, filename=None):
    """
        Transforms matrix to libfm format
    :param X: numpy.array
        data
    :param y: numpy.array
        predicted variable
    :param filename: str
        Output file name
    :return:
    """

    def row2libfm(x):
        return "{label} {data}\n".format(label=x[-1],
                                         data=" ".join(["{0}:{1}".format(k + 1, v) for k, v in enumerate(x[:-1]) if v != 0]))

    if y is None:
        output = map(row2libfm, np.hstack((X, np.zeros((X.shape[0], 1)))))
    else:
        output = map(row2libfm, np.hstack((X, y.reshape((-1, 1)))))

    # print len(output), output

    with open(filename, "w") as f:
        f.writelines(output)


def libfm_meta_file(filename, groups_size):
    """
        Creates Meta file with groupID for each featureID,
        see http://www.libfm.org/libfm-1.40.manual.pdf for details
    :param filename: str
        Output file name
    :param groups_size: list
        List of number of columns for each group
    """
    total_size = sum(groups_size)
    meta = np.zeros((total_size, 1), dtype=np.int8)
    ix_start = 0
    ix_end = 0
    for i, n in enumerate(groups_size):
        ix_end += n
        meta[ix_start:ix_end] = i
        ix_start = ix_end
    np.savetxt(filename, meta, fmt="%d")


ENCHANT_DICT = enchant.Dict("en_US")
STEMMER = PorterStemmer()


def kappa(y_true, y_pred, weights=None, allow_off_by_one=False):
    """
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.

    In the course of calculating kappa, all items in `y_true` and `y_pred` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :param weights: Specifies the weight matrix for the calculation.
                    Options are:

                        -  None = unweighted-kappa
                        -  'quadratic' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimensional numpy array = a custom matrix of
                           weights. Each weight corresponds to the
                           :math:`w_{ij}` values in the wikipedia description
                           of how to calculate weighted Cohen's kappa.

    :type weights: str or numpy array
    :param allow_off_by_one: If true, ratings that are off by one are counted as
                             equal, and all other differences are reduced by
                             one. For example, 1 and 2 will be considered to be
                             equal, whereas 1 and 3 will have a difference of 1
                             for when building the weights matrix.
    :type allow_off_by_one: bool
    """
    logger = logging.getLogger(__name__)

    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    try:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError as e:
        logger.error("For kappa, the labels should be integers or strings "
                     "that can be converted to ints (E.g., '4.0' or '3').")
        raise e

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, string_types):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('Invalid weight scheme specified for '
                                     'kappa: {}'.format(wt_scheme))

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k

scorer = make_scorer(kappa, greater_is_better=True, weights="quadratic")


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, alpha=0.7, correction=True, bandwidth=0.25):
        self.columns = columns
        self.alpha = alpha
        self.correction = correction
        self.bandwidth = bandwidth

    def __word_processor(self, word):
        #     if not ENCHANT_DICT.check(word):
        #         suggestions = ENCHANT_DICT.suggest(word)
        #         word_ = suggestions[0].lower() if suggestions else word
        #     else:
        #         word_ = word

        #     word_ = STEMMER.stem(word_.split("'")[0])
        return word.replace("-", "")

    def __correction(self, r):
        if self.correction:
            return r if r > 0.5 else 0
        else:
            return r

    def __matching_rate(self, x):
        wordsProcessor = np.vectorize(self.__word_processor)
        tokens = self.tokenizer(x["query"])
        query = wordsProcessor(tokens) if tokens else None

        tokens = self.tokenizer(x["product_title"])
        product_title = wordsProcessor(tokens) if tokens else None

        tokens = self.tokenizer(x["product_description"])
        product_descr = wordsProcessor(tokens) if tokens else None

        title_edit_dist = title_intersection = -1
        descr_edit_dist = descr_intersection = -1
        all_edit_dist = all_intersection = -1

        n = len(query)
        query_set = set(query)
        if product_title is not None:
            product_title_set = set(product_title)
            query_title_ratio = [self.__correction(max([SequenceMatcher(None, w, w1).ratio() for w1 in product_title])) for w in query]
            title_edit_dist = float(sum(query_title_ratio)) / n
            title_intersection = float(len(query_set.intersection(product_title_set))) / n

        if product_descr is not None:
            product_descr_set = set(product_descr)
            query_descr_ratio = [self.__correction(max([SequenceMatcher(None, w, w1).ratio() for w1 in product_descr])) for w in query]
            descr_edit_dist = float(sum(query_descr_ratio)) / n
            descr_intersection = float(len(query_set.intersection(product_descr_set))) / n

        if product_title is not None and product_descr is not None:
            ratio = [max(x1, x2) for x1, x2 in zip(query_title_ratio, query_descr_ratio)]
            all_edit_dist = float(sum(ratio)) / n
            all_intersection = float(len(query_set.intersection(product_descr_set.union(product_title_set)))) / n
        elif product_title is not None:
            all_edit_dist = title_edit_dist
            all_intersection = title_intersection
        elif product_descr is not None:
            all_edit_dist = descr_edit_dist
            all_intersection = descr_intersection

        title_similarity = self.alpha * title_intersection + (1 - self.alpha) * title_edit_dist
        descr_similarity = self.alpha * descr_intersection + (1 - self.alpha) * descr_edit_dist
        all_similarity = self.alpha * all_intersection + (1 - self.alpha) * all_edit_dist
        return pd.Series(dict(
                              # title_edit_dist=title_edit_dist,
                              # descr_edit_dist=descr_edit_dist,
                              # title_intersection=title_intersection,
                              # descr_intersection=descr_intersection,
                              all_edit_dist=all_edit_dist,
                              all_intersection=all_intersection,
                              all_similarity=all_similarity,
                              title_similarity=title_similarity,
                              descr_similarity=descr_similarity
                              ))

    def fit(self, X, y):
        '''

        :param X: numpy.array
        :return: self
        '''
        data = pd.DataFrame(X, columns=self.columns)
        self.tokenizer = CountVectorizer(stop_words="english", lowercase=True,
                                         strip_accents="unicode", token_pattern=r"(?u)\b\w\w+-?\w*\b").build_analyzer()

        # rates = data.apply(self.__matching_rate, axis=1).values.astype(np.uint8)
        # self.ratesEncoder = OneHotEncoder(dtype=np.int8, sparse=False, handle_unknown="ignore")
        # self.ratesEncoder.fit(rates)

        __ = pd.concat([data["query"], pd.DataFrame(y, columns=["median_relevance"])], axis=1).groupby(["query", "median_relevance"])["query"].count()
        _ = data["query"].groupby(data["query"]).count()
        ratio = __.divide(_, level=0).unstack().fillna(0)

        clusterer = MeanShift(bandwidth=self.bandwidth)
        labels = clusterer.fit_predict(ratio.values)

        self.query2cluster = dict(zip(ratio.index.values, labels))
        self.n_query_clusters = np.unique(labels).size
        self.queryEncoder = OneHotEncoder(dtype=np.bool, sparse=False, handle_unknown="ignore")
        self.queryEncoder.fit(labels.reshape((-1, 1)))

        # self.titleEncoder = CountVectorizer(stop_words="english", lowercase=True, strip_accents="unicode", ngram_range=(1, 4), min_df=50, max_df=3000, binary=True, dtype=np.bool)
        # self.titleEncoder.fit(data["product_title"].values)
        #
        # self.descriptionEncoder = CountVectorizer(stop_words="english", lowercase=True, strip_accents="unicode", ngram_range=(1, 4), min_df=50, max_df=3000, binary=True, dtype=np.bool)
        # self.descriptionEncoder.fit(data["product_description"].values)
        return self

    def transform(self, X, y=None):
        '''

        :param data: numpy.array
        :return: numpy.array
        '''
        data = pd.DataFrame(X, columns=self.columns)
        data_rates = data.apply(self.__matching_rate, axis=1).values
        # data_rates = self.ratesEncoder.transform(rates)
        labels = np.vectorize(lambda x: self.query2cluster[x])(data["query"].values)

        data_query = self.queryEncoder.transform(labels.reshape((-1, 1)))
        # data_query = self.queryEncoder.transform(np.vectorize(lambda x: abs(hash(x)) % 30000)(data["query"].values.reshape((-1, 1))))
        # data_title = np.asarray(self.titleEncoder.transform(data["product_title"].values).todense())
        # data_description = np.asarray(self.descriptionEncoder.transform(data["product_description"].values).todense())
        self.groups_size = [data_rates.shape[1], data_query.shape[1]]
        return np.hstack((data_rates, data_query))


def load_data(filename, has_label=True):
    start = time.time()
    data = pd.read_csv(filename, na_filter=False)

    id = data["id"].values
    data.drop(["id"], axis=1, inplace=True)

    if has_label:
        relevance = data[["median_relevance", "relevance_variance"]].copy()
        data.drop(["median_relevance", "relevance_variance"], axis=1, inplace=True)

    print "transformation time: ", time_passed(start)
    return (data, id, relevance) if has_label else (data, id)


class FMClassifier(BaseEstimator):
    def __init__(self, task="r", train="data/train.libfm", test="data/validation.libfm", meta="data/libfm.meta",
                 dim="'1,1,1'", iter_=1000, method="mcmc", init_stdev=0.1, out="prediction.txt"):
        self.task = task
        self.train = train
        self.test = test
        self.meta = meta
        self.dim = dim
        self.iter_ = iter_
        self.method = method
        self.init_stdev = init_stdev
        self.out = out

    def fit(self, train, y_train, test):
        self.classes = np.unique(y_train)
        for c in self.classes:
            y_train_binary = (y_train == c).astype(np.int8)
            numpy2libfm(X=train, y=y_train_binary, filename=self.train)
            numpy2libfm(X=test, y=None, filename=self.test)

            params = dict(task=self.task, train=self.train, test=self.test, meta=self.meta,
                          dim=self.dim, iter=self.iter_, method=self.method, init_stdev=self.init_stdev,
                          out="{0}_{1}".format(self.out, c))

            bash_params = ["-{0} {1}".format(k, v) for k, v in params.iteritems()]
            bash_command = "./libFM {0}".format(" ".join(bash_params))
            process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, shell=True)
            output = process.communicate()[0]

    def predict_proba(self, X=None):
        pred = []
        for c in self.classes:
            pred.append(np.loadtxt("{0}_{1}".format(self.out, c)))
        pred = np.array(pred).transpose()
        pred = pred / pred.sum(axis=1).reshape((-1, 1)).astype(float)  # normalize, so sum(probs over classes) = 1

        return pred

    def predict(self, X=None):
        pred = []
        for c in self.classes:
            pred.append(np.loadtxt("{0}_{1}".format(self.out, c)))
        pred = np.array(pred).transpose()
        pred = pred / pred.sum(axis=1).reshape((-1, 1)).astype(float)  # normalize, so sum(probs over classes) = 1

        return np.argmax(pred, 1) + 1

if __name__ == "__main__":

    data, id_, relevance = load_data("data/train.csv", has_label=True)

    # dataTransformer = DataTransformer(alpha=0.7, correction=True, bandwidth=0.25)
    # data_preprocessed = dataTransformer.fit_transform(data, relevance.median_relevance)
    # print data_preprocessed.shape

    # Model
    cl = Pipeline([("t", DataTransformer(columns=data.columns.values, alpha=0.7, correction=True, bandwidth=0.125)),
                  ("e", GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, subsample=0.5, random_state=0))])

    param_grid = dict(e__n_estimators=[150], e__max_depth=[2, 3, 4],
                      e__min_samples_leaf=[10, 20, 30], e__subsample=[0.5, 0.8])
    grid_search = GridSearchCV(cl, param_grid=param_grid, scoring=scorer, cv=6, verbose=True, n_jobs=6)
    grid_search.fit(data.values, relevance.median_relevance.values)

    print grid_search.grid_scores_
    print grid_search.best_params_, grid_search.best_score_
    print "train score", scorer(grid_search.best_estimator_, data.values, relevance.median_relevance.values)

    print "n query groups:", grid_search.best_estimator_.steps[0][-1].n_query_clusters
    print "actual relevance distr:", np.bincount(relevance.median_relevance.values) / float(relevance.median_relevance.values.size)
    prediction = grid_search.best_estimator_.predict(data.values)
    print "pred relevance distr:", np.bincount(prediction) / float(prediction.size)

    print grid_search.best_estimator_.steps[-1][-1].feature_importances_

    # SUBMISSION
    test_data, id_ = load_data("data/test.csv", has_label=False)
    # test_data_preprocessed, _ = dataTransformer.transform(test_data)

    prediction = grid_search.best_estimator_.predict(test_data.values)
    pred = pd.DataFrame({"id": id_, "prediction": prediction})
    pred.to_csv("submission.txt", index=False)

