__author__ = 'alfiya'

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV, ParameterGrid
import logging
from six import string_types
from sklearn.metrics import make_scorer, confusion_matrix
import enchant
from nltk.stem.porter import PorterStemmer
from difflib import SequenceMatcher
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.cluster import MeanShift, KMeans
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
import time
import subprocess
import csv
from sklearn.decomposition import NMF, TruncatedSVD
from scipy.spatial.distance import cosine
import cPickle
import re
import inspect
from sympy.utilities.lambdify import lambdify
from sympy.abc import x
np.set_printoptions(suppress=True)
pd.set_option('max_colwidth',100)
from bs4 import BeautifulSoup
import datetime

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

tmp_set = []
def word_processor(word):
    #     if not ENCHANT_DICT.check(word):
    #         suggestions = ENCHANT_DICT.suggest(word)
    #         word_ = suggestions[0].lower() if suggestions else word
    #     else:
    #         word_ = word

    word_ = STEMMER.stem(word.split("'")[0])
    # if re.findall("-", word):
    #     if not re.findall("[0-9]", word):
    #         tmp_set.append(word)
    #         if min(len(_) for _ in word.split("-")) > 3:
    #             out = word.split("-")
    #         else:
    #             out = word.replace("-", "")
    #     else:
    #         out = word.split("-")
    #
    # else:
    #     out = word

    return word_

wordsProcessor = np.vectorize(word_processor)
analyzer = CountVectorizer(stop_words="english", lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                           token_pattern=r"(?u)\b\w\w+\b").build_analyzer()      #r"(?u)\b\w+-?\w\w*-?\w*-?\w*-?\w*\b"
tokenizer = CountVectorizer(stop_words="english", lowercase=True, strip_accents="unicode", ngram_range=(1, 1),
                            token_pattern=r"(?u)\b\w\w+\b").build_analyzer()      #r"(?u)\b\w+-?\w\w*-?\w*-?\w*-?\w*\b" # def tokenizer(x):
def tokenize(x):
    tokens = tokenizer(x)
    out = []
    for t in tokens:
        word = word_processor(t)
        if type(word) == list:
            out.extend(word)
        else:
            out.append(word)
    return np.array(out)

class QueryProductMatch(BaseEstimator, TransformerMixin):
    """Matching rates between query and product
    Consists of the followting steps:

    Tokenization of text
    -------------------------------------------------------------------------------------
    Applied to fields "query", "product_description", "product_title"
    Params:
        tokenizer: function
            tokenizer
        wordsProcessor: function (vectorized by np.vectorize)
            applied to each word after tokenization

    Matching rates between query and product
    -------------------------------------------------------------------------------------
    Calculates
        tokens intersection between query and product
        maximum edit similarity
        linear combination between intersection rate and edit similarity rate
    Params:
        alpha: float between 0 and 1
            used in weighted sum of intersection_rate and edit similarity rate
            similarity = alpha * intersection_rate + (1-alpha) * edit_similarity_rate
        correction: bool
            whether to apply correction function on edit distance rate or not

    Additional params:
    ------------------
        columns: list or np.array
            column names for data
    """
    def __init__(self, columns, tokenizer, alpha=0.7, correction=True, edit_rate_threshold=0.5):
        self.columns = columns
        self.tokenizer = tokenizer

        self.alpha = alpha
        self.correction = correction
        self.edit_rate_threshold = edit_rate_threshold

    def correct_rate(self, r):
        if self.correction:
            return r if r > self.edit_rate_threshold else 0
        else:
            return r

    def matching_rate(self, x):
        query = x["query_tokens"]
        product_title = x["title_tokens"]
        product_descr = x["descr_tokens"]

        title_edit_dist = title_intersection = -1
        descr_edit_dist = descr_intersection = -1
        all_edit_dist = all_intersection = -1

        n = len(query)
        query_set = set(query)
        product_title_set = set()
        product_descr_set = set()
        irrelevant_words = 0

        if product_title.size:
            product_title_set = set(product_title)
            # if x["query"] in self.irrel_tokens:
            #     irrelevant_words = len(product_title_set.intersection(self.irrel_tokens[x["query"]]))

            title_intersection = float(len(query_set.intersection(product_title_set))) / n
            if title_intersection == 1:
                query_title_ratio = [1] * n
                title_edit_dist = 1
            else:
                query_title_ratio = [self.correct_rate(max([SequenceMatcher(None, w, w1).ratio() for w1 in product_title])) for w in query]
                title_edit_dist = float(sum(query_title_ratio)) / n

        if product_descr.size:
            product_descr_set = set(product_descr)
            descr_intersection = float(len(query_set.intersection(product_descr_set))) / n
            if descr_intersection == 1:
                query_descr_ratio = [1] * n
                descr_edit_dist = 1
            else:
                query_descr_ratio = [self.correct_rate(max([SequenceMatcher(None, w, w1).ratio() for w1 in product_descr])) for w in query]
                descr_edit_dist = float(sum(query_descr_ratio)) / n

        if product_title.size and product_descr.size:
            ratio = [max(x1, x2) for x1, x2 in zip(query_title_ratio, query_descr_ratio)]
            all_edit_dist = float(sum(ratio)) / n
            all_intersection = float(len(query_set.intersection(product_descr_set.union(product_title_set)))) / n
        elif product_title.size:
            all_edit_dist = title_edit_dist
            all_intersection = title_intersection
        elif product_descr.size:
            all_edit_dist = descr_edit_dist
            all_intersection = descr_intersection

        title_similarity = self.alpha * title_intersection + (1 - self.alpha) * title_edit_dist
        descr_similarity = self.alpha * descr_intersection + (1 - self.alpha) * descr_edit_dist
        all_similarity = self.alpha * all_intersection + (1 - self.alpha) * all_edit_dist

        return pd.Series([all_edit_dist, all_intersection, all_similarity, title_similarity, descr_similarity],
                         index="all_edit_dist all_intersection all_similarity title_similarity descr_similarity".split())

    def tokenize(self, x):
        x["query_tokens"] = self.tokenizer(x["query"])
        # x["query_tokens"] = self.wordsProcessor(tokens) if tokens else np.array([])

        x["title_tokens"] = self.tokenizer(x["product_title"])
        # x["title_tokens"] = self.wordsProcessor(tokens) if tokens else np.array([])

        x["descr_tokens"] = self.tokenizer(x["product_description"])
        # x["descr_tokens"] = self.wordsProcessor(tokens) if tokens else np.array([])
        x = x.drop(["product_title", "product_description"])
        return x

    def fit(self, X, y):
        """

        :param X: numpy.array
        :return: self
        """
        # data = pd.DataFrame(X, columns=self.columns)
        #
        # # TOKENIZE
        # data = data.apply(self.tokenize, axis=1)
        #
        # # relevant product title for each query
        # rel = y > 2
        # rel_tokens = dict()
        # for q, t in data[["query", "title_tokens"]][rel].itertuples(index=False):
        #     if q not in rel_tokens:
        #         rel_tokens[q] = set()
        #     rel_tokens[q].update(t)
        #
        # # irrelevant product title for each query
        # self.irrel_tokens = dict()
        # for q, t in data[["query", "title_tokens"]][~rel].itertuples(index=False):
        #     if q not in self.irrel_tokens:
        #         self.irrel_tokens[q] = set()
        #     self.irrel_tokens[q].update(t)
        #
        # for q in self.irrel_tokens:
        #     if q in rel_tokens:
        #         self.irrel_tokens[q].difference_update(rel_tokens[q])

        return self

    def transform(self, X, y=None):
        """

        :param data: numpy.array
        :return: numpy.array
        """

        data = pd.DataFrame(X, columns=self.columns)

        # TOKENIZE
        data = data.apply(self.tokenize, axis=1)

        # MATCHING RATES
        data_rates = data.apply(self.matching_rate, axis=1).values

        return data_rates

class QueryClustering(BaseEstimator, TransformerMixin):
    """
    Query clustering
    -------------------------------------------------------------------------------------
    Cluster queries according to their distribution of relevance
    (so queries with similar relevance score distribution should be in the same cluster)
    Params:
        query_clusterer: str, default="MeanShift"
            clustering algorithm to use
            must be in globals()
        clusterer_params: dict
            default=dict(bandwidth=0.125)


    Additional params:
    ------------------
        columns: list or np.array
            column names for data
    """
    def __init__(self, columns, query_clusterer=None, clusterer_params=None):
        self.columns = columns

        self.query_clusterer = query_clusterer
        self.clusterer_params = clusterer_params

    def fit(self, X, y):
        """

        :param X: numpy.array
        :return: self
        """

        data = pd.DataFrame(X, columns=self.columns)

        # QUERY CLUSTERING
        __ = pd.concat([data["query"], pd.DataFrame(y, columns=["median_relevance"])], axis=1).groupby(["query", "median_relevance"])["query"].count()
        _ = data["query"].groupby(data["query"]).count()
        ratio = __.divide(_, level=0).unstack().fillna(0)

        if self.query_clusterer is None:
            labels = np.arange(ratio.shape[0])
        else:
            clusterer = globals()[self.query_clusterer](**self.clusterer_params)
            labels = clusterer.fit_predict(ratio.values)

        self.query2cluster = dict(zip(ratio.index.values, labels))
        self.n_query_clusters = np.unique(labels).size
        self.queryEncoder = OneHotEncoder(dtype=np.bool, sparse=False, handle_unknown="ignore")
        self.queryEncoder.fit(labels.reshape((-1, 1)))

        return self

    def transform(self, X, y=None):
        """

        :param data: numpy.array
        :return: numpy.array
        """

        data = pd.DataFrame(X, columns=self.columns)

        # QUERY CLUSTERS
        labels = np.vectorize(lambda x: self.query2cluster[x])(data["query"].values)
        data_query = self.queryEncoder.transform(labels.reshape((-1, 1)))

        return data_query

class TopicModel(BaseEstimator, TransformerMixin):
    """
    Topic modeling and semantic similarity
    -------------------------------------------------------------------------------------
    Builds TF-IDF matrix for PRODUCTS using TfIdfVectorizer from sklearn
    Applies matrix decomposition on tf-idf matrix
    and calculates semantic similarity between query and product using cosine similarity
    Params:
        params for TfIdfVectorizer:
        (see docs for sklearn.feature_extraction.text.TfidfVectorizer)
            min_df
            max_df
            ngram_range
            smooth_idf

        topics_model: str, default="NMF"
            method for matrix decomposition
            should be in globals()
        topics_params: dict
            params to pass in topics_model
            default=dict(n_components=2, init='nndsvd', sparseness="components")

    Additional params:
    ------------------
        columns: list or np.array
            column names for data
        use_semantic_sim: bool
            whether to include semantic_similarity measure in output data
        use_topics: bool
            whether to include topics (transformed tf-idf matrix) in output data
    """
    def __init__(self, columns, tfidf_params=None, topics_model="NMF", topics_params=None,
                 use_semantic_sim=True, use_topics=False):
        self.columns = columns

        self.tfidf_params = tfidf_params

        self.topics_model = topics_model
        self.topics_params = topics_params

        self.use_semantic_sim = use_semantic_sim
        self.use_topics = use_topics

    def fit(self, X, y):
        """

        :param X: numpy.array
        :return: self
        """

        # one of the flags @use_semantic_sim or @use_topics should be True
        if not (self.use_semantic_sim or self.use_topics):
            raise Exception("One of the flags @use_semantic_sim or @use_topics should be True")

        # Set default
        if self.topics_params is None:
            self.topics_params = dict(n_components=2, init='nndsvd', sparseness="components")
        if self.tfidf_params is None:
            self.tfidf_params = dict(min_df=5, max_df=1., ngram_range=(1, 1), norm="l2", use_idf=True, smooth_idf=True)

        # TOPIC MODELING FOR PRODUCTS
        # if self.fit_topics:
        self.tfIdfVectorizer = TfidfVectorizer(stop_words='english', **self.tfidf_params)
        tf_idf = self.tfIdfVectorizer.fit_transform(PRODUCTS)

        start = time.time()
        self.topicsExtractor = globals()[self.topics_model](**self.topics_params)
        self.product_topics = self.topicsExtractor.fit_transform(tf_idf)
        self.query_topics = self.topicsExtractor.transform(self.tfIdfVectorizer.transform(QUERIES))
        print "Tokens num: {0} Decomposition time: {1} Variance {2}".\
            format(tf_idf.shape[1], time_passed(start),sum(self.topicsExtractor.explained_variance_ratio_))

        return self

    def transform(self, X, y=None):
        """

        :param data: numpy.array
        :return: numpy.array
        """

        data = pd.DataFrame(X, columns=self.columns)

        out = []
        # SEMANTIC SIMILARITY
        if self.use_semantic_sim:
            semantic_similarity = data.apply(cosine_sim, axis=1, args=(self.query_topics, self.product_topics))
            out.append(semantic_similarity.values.reshape((-1, 1)))

        # TOPICS
        if self.use_topics:
            data_topics = np.array(map(lambda x: self.product_topics[PRODUCTS == x].ravel(), data["product"].values.reshape((-1, 1))))
            out.append(data_topics)
        return np.hstack(out)


def load_data(filename, has_label=True):
    start = time.time()
    separ = re.compile(r"\s+")
    link = re.compile(r"http://[A-Za-z0-9\.\-\?_&=%/#]+")  # \S
    css_rem = re.compile(r"<.*\}")  # ([\.#]\w+\s?|\w\w\s?)?

    def clear_string(s):
        s = re.sub(separ, " ", s)  # replace separators to " "
        s = re.sub(css_rem, " __css__ ", s)  # remove css
        s = re.sub(link, " __link__ ", s)  # replace all http links to special word
        s = BeautifulSoup(s).get_text()  # parse html
        s = re.sub(r" +", " ", s)  # remove extra spaces

        return s
    # ((\.[\w]+|\w\w)[\s,]*)+\{.*?\}
    data = pd.read_csv(filename, na_filter=False)

    id = data["id"].values
    data.drop(["id"], axis=1, inplace=True)

    if has_label:
        relevance = data[["median_relevance", "relevance_variance"]].copy()
        data.drop(["median_relevance", "relevance_variance"], axis=1, inplace=True)

    data["product_title"] = data["product_title"].apply(clear_string)
    data["product_description"] = data["product_description"].apply(clear_string)
    data["product"] = data.apply(lambda row: ". ".join(row[["product_title", "product_description"]]), axis=1)
    print "load time: ", time_passed(start)
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


def cosine_sim(x, query_topics, topics):
    x1 = query_topics[QUERIES == x["query_"]]
    x2 = topics[PRODUCTS == x["product"]]
    sim = 1 - cosine(x1, x2)
    return 0. if np.isnan(sim) else sim

class GridSearch(BaseEstimator):
    """GridSearch
    Supposed that estimator is a Pipeline with at least 2 steps:
        data tranformation step - performed before cross validation
        middle transformations - fitted and transformed as is
        modeling step - passed on sklearn.cross_val_score

    More time effective in comparison to sklearn.GridSearchCV
    because data transformation step runs 1 time for each set of parameters (instead of @cv times)
    """
    def __init__(self, estimator, param_grid, scoring, cv=3, n_jobs=1, verbose=True, refit=True, acceptable_overfit=1.):
        """

        :param estimator: sklearn.Pipeline class
        :param param_grid: dict
            param grid as in sklearn.GridSearchCV
        :param scoring: function
            scorer function as in sklearn.GridSearchCV
        :param cv: int
            number of folds
        :param n_jobs: int
            number of jobs to run in parallel
        :param verbose: bool
            level of verbosity
        :param refit: bool
            Whether to fit the model with the best params on a whole dataset or not
        :param acceptable_overfit: float
            Acceptable diff between train and validation scores
        :return:
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit = refit
        self.acceptable_overfit=acceptable_overfit

    def fit_topics(self, params, est):
        """
        Decides whether to fit topics model or not
        Matrix decomposition is time consuming, that's why I implemented this step
        :param params:
        :return:
        """
        t_attr_names = [name for name, value in inspect.getmembers(est.steps[0][-1], lambda x: not(inspect.isroutine(x)))]
        t_name = est.steps[0][0]
        if "topicsExtractor" not in t_attr_names:
            fit_topics = True
        else:
            t_params = " ".join([k for k in params if k.split("__")[0] == t_name])
            fit_topics = bool(re.findall("topic", t_params))
        est.set_params(**{"__".join([t_name, "fit_topics"]): fit_topics})

    def middle_transformations(self, est, X, y):
        if len(est.steps) > 2:
            tmp = Pipeline([(name, obj) for name, obj in est.steps[1:-1]] + [("dummy", DummyClassifier())])
            transformed_data, fit_params = tmp._pre_transform(X, y)
            return transformed_data
        else:
            return X

    def fit(self, X, y):

        self.grid_scores_ = []
        self.best_estimator_, self.train_score, self.train_test_diff, self.train_test_diff_p = None, None, None, None

        grid_len = len(ParameterGrid(self.param_grid))
        print "Grid Search over {0} params sets with cv {1}, total {2} fits".\
            format(grid_len, self.cv, grid_len * self.cv)

        transformed_data = None
        for params in ParameterGrid(self.param_grid):
            start = time.time()
            # e = clone(self.estimator)
            self.estimator.set_params(**params)
            # self.fit_topics(params, self.estimator)

            # transform data
            if [k for k in params if k.split("__")[0] == "t"] or transformed_data is None:
                transformed_data = self.estimator.steps[0][-1].fit_transform(X, y)
                transformed_data = self.middle_transformations(self.estimator, transformed_data, y)

            # cross validation
            scores = cross_val_score(self.estimator.steps[-1][-1], transformed_data, y, scoring=self.scoring,
                                     cv=self.cv, n_jobs=self.n_jobs)

            self.estimator.steps[-1][-1].fit(transformed_data, y)
            train_score = self.scoring(self.estimator.steps[-1][-1], transformed_data, y)
            self.grid_scores_.append({"params": params,
                                      "score": {"mean": round(np.mean(scores), 5),
                                                "median": round(np.median(scores), 5),
                                                "std": round(np.std(scores), 5)},
                                      "overfitting": train_score - np.mean(scores)})
            if self.verbose:
                print "params: {1}\nscore: {2} overfit: {3}\ntime passed: {0} time stamp: {4}\n".\
                    format(time_passed(start), params, self.grid_scores_[-1]["score"], self.grid_scores_[-1]["overfitting"], datetime.datetime.now())

        max_score = max([v["score"]["mean"] for v in self.grid_scores_
                         if v["overfitting"] <= self.acceptable_overfit])
        idx = [i for i, v in enumerate(self.grid_scores_)
               if v["overfitting"] <= self.acceptable_overfit and v["score"]["mean"] == max_score][0]
        self.best_score_ = self.grid_scores_[idx]["score"]
        self.best_params_ = self.grid_scores_[idx]["params"]

        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            # self.fit_topics(self.best_params_, self.best_estimator_)

            transformed_data = self.best_estimator_.steps[0][-1].fit_transform(X, y)
            transformed_data = self.middle_transformations(self.best_estimator_, transformed_data, y)
            self.best_estimator_.steps[-1][-1].fit(transformed_data, y)

            self.train_score = self.scoring(self.best_estimator_.steps[-1][-1], transformed_data, y)
            self.train_test_diff = self.train_score - self.best_score_["mean"]
            self.train_test_diff_p = self.train_test_diff * 100 / self.train_score

        if self.verbose:
            print '\nbest params: {0}\nbest scores: {1}'.format(self.best_params_, self.best_score_)
            if self.refit:
                print "train test diff: {0}, train test diff %: {1}".\
                    format(round(self.train_test_diff, 5), round(self.train_test_diff_p, 3))

    def predict(self, X):
        if self.best_estimator_ is None:
            raise Exception("Best estimator has not been fitted, use @refit=True to fix it")
        else:
            return self.best_estimator_.predict(X)

# def hyphen_words_correction(row):
#     for w in hyphen_words_replace:
#         row["query"] = w[0].sub(w[1], row["query"])
#         row["product_title"] = w[0].sub(w[1], row["product_title"])
#         row["product_description"] = w[0].sub(w[1], row["product_description"])
#         # row["product"] = ". ".join(row[["product_title", "product_description"]])
#         row["product"] = w[0].sub(w[1], row["product"])
#     return row


if __name__ == "__main__":
    # LOAD
    data, id_0, relevance = load_data("data/train.csv", has_label=True)
    test_data, id_ = load_data("data/test.csv", has_label=False)

    # PRODUCTS = pd.concat([data["product"], test_data["product"]], ignore_index=True).drop_duplicates().values
    l_f = lambda x: " ".join(["t_" + v for v in tokenizer(x["product_title"])]) \
                    + " " + " ".join(["d_" + v for v in tokenizer(x["product_description"])]) #\
                    # + " " + " ".join(["q_" + v for v in tokenizer(x["query"])])
    data["product"] = data.apply(l_f, axis=1)
    test_data["product"] = test_data.apply(l_f, axis=1)
    PRODUCTS = pd.concat([data["product"], test_data["product"]], ignore_index=True).drop_duplicates().values

    l_f = lambda x: " ".join(["t_" + v for v in tokenizer(x)])  # + " " + " ".join(["d_" + v for v in tokenizer(x)])
    data["query_"] = data["query"].apply(l_f)
    test_data["query_"] = test_data["query"].apply(l_f)
    QUERIES = pd.concat([data["query_"], test_data["query_"]], ignore_index=True).drop_duplicates().values

    print PRODUCTS[:3]
    # MODEL

    dataTransformer = FeatureUnion([
        ("m", QueryProductMatch(columns=data.columns.values, tokenizer=tokenize, edit_rate_threshold=0.25, alpha=0.5)),
        ("q", QueryClustering(columns=data.columns.values, query_clusterer=None,
                              clusterer_params=None)),  # dict(n_clusters=8, random_state=10)
        ("t", TopicModel(columns=data.columns.values, use_semantic_sim=True, use_topics=True,
                         tfidf_params=dict(min_df=5, max_df=1., ngram_range=(1, 2), norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True),
                         topics_model="TruncatedSVD", topics_params=dict(n_components=200, n_iter=5, random_state=10)))
    ])
    # estimator = SVC(C=31.622776601683793, kernel='poly', degree=4, gamma=0.0, coef0=1.0, shrinking=True,
    #                 probability=False, tol=0.001, class_weight="auto",
    #                 verbose=False, max_iter=2000000, random_state=0)
     #RandomForestClassifier(n_estimators=150, max_depth=11, min_samples_leaf=10, random_state=0)
    estimator = SVC(C=10, kernel='rbf', gamma=0.0, coef0=0.0, shrinking=True,
                    probability=False, tol=0.001, class_weight=None,
                    verbose=False, max_iter=2000000, random_state=0)
    # estimator = RandomForestClassifier(n_estimators=150, max_depth=11, min_samples_leaf=10, random_state=0)
    cl = Pipeline([("t", dataTransformer),
                   ("s", StandardScaler()),
                   ("e", estimator)])

    param_grid = dict()
                  # dict(e__C=np.logspace(-2, 2, 5), e__kernel=['rbf'],
                  #      e__degree=[4], e__gamma=np.logspace(-3, -1, 5))]

    grid_search = GridSearchCV(cl, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=1,
                               verbose=True, refit=True)  # , acceptable_overfit=0.5
    grid_search.fit(data.values, relevance.median_relevance.values)

    print "actual relevance distr:", np.bincount(relevance.median_relevance.values) / float(relevance.median_relevance.values.size)
    prediction = grid_search.predict(data.values)
    print "pred relevance distr:", np.bincount(prediction) / float(prediction.size)
    print grid_search.grid_scores_
    print "overfit", kappa(relevance.median_relevance.values, prediction, weights="quadratic") - grid_search.best_score_
    cPickle.dump(grid_search.best_estimator_.steps[0][-1].transform(data.values), open("data.pkl", "w"))
    # cPickle.dump(grid_search, open("grid_search.pkl", "w"))

    # SUBMISSION
    prediction = grid_search.predict(test_data.values)
    pred = pd.DataFrame({"id": id_, "prediction": prediction})
    pred.to_csv("submission.txt", index=False)
# mean: 0.65842, std: 0.01080