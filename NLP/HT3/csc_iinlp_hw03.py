"""
Homework #3

Implement an n-gram language model that can separate true sentences from the artificially obtained sentences.

"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


ALPHA = 0.01

class LanguageModel(object):
    """
        n-gramm model
        You can improve and extend the model with any methods, that were on lection
    """

    def __init__(self, ngram_size=2):

        if ngram_size < 2:
            raise Exception

        # n-gramms size, model param
        self.ngram_size = ngram_size

        # the simple way to build n-gramms
        self.vectorizer = CountVectorizer(ngram_range=(ngram_size, ngram_size), analyzer='word')

        # the simple way to build contexts (on what we will divide during probabilities estimation)
        self.context_vectorizer = CountVectorizer(ngram_range=(ngram_size - 1, ngram_size - 1), analyzer='word')
        self.ngram_counts = defaultdict(lambda: -np.inf)
        self.words_set_size = None

    def fit(self, sentences):
        """
            Model training on sentence-splitted text
            :param sentences: the list of sentences
        """

        print("Fitting sentences")
        # the matrix of n-gramms counters
        counts_matrix = self.vectorizer.fit_transform(sentences)
        # ...and contexts
        counts_context_matrix = self.context_vectorizer.fit_transform(sentences)
        # total number of words
        self.dictionary = set([key for ngram in self.vectorizer.vocabulary_.keys() for key in ngram.split(" ")])
        self.words_set_size = len(self.dictionary)
        print(self.dictionary)
        print("Summing...")
        # count the number of n-gramms and contexts as sum of all columns
        # Count(x1,x2,...xn)
        sum_ngram = np.sum(counts_matrix, axis=0).A1
        # Count(x1,x2,...x_{n-1})
        sum_context = np.sum(counts_context_matrix, axis=0).A1
        V = len(sum_context)

        print("shapes: ", sum_ngram.shape, sum_context.shape)
        print("Iterating through ngrams...")

        for ngram in self.vectorizer.vocabulary_:
            # build the context by removing the last word
            words = " ".join(ngram.split(" ")[:-1])

            # the number of n-gramms = ngram
            id = (self.vectorizer.vocabulary_.get(ngram))
            total = sum_ngram[id]

            # the number of contexts = words
            id = (self.context_vectorizer.vocabulary_.get(words))
            total_ctx = sum_context[id]

            # calculate the log of n-gramm probability
            logprob = np.log2((total + ALPHA) / (total_ctx + ALPHA * V))

            # do updates
            self.ngram_counts[ngram] = logprob

        # the log of n-gramm probability for UNKNOWN N-GRAMMS
        # TODO: your code here
        self.ngram_counts.default_factory = lambda: np.log2(1 / V)

        return self

    def __log_prob_sentence(self, sentence):
        """
            Evaluate the log of sentence probability with frequences in terms of current model
        :param sentence:
        :return:
        """

        # split on tokens
        splitted = sentence.split(" ")
        sum_log = 1.0

        for i in range(len(splitted) - self.ngram_size + 1):
            # take the next n-gramm
            ngram = " ".join(splitted[i: i + self.ngram_size])

            # if it is unknown, then return log of zero
            # otherwise sum the logs of probabilities
            sum_log += self.ngram_counts[ngram]

        return sum_log

    def log_prob(self, sentence_list):
        """
            The log of probability of every sentence in the list
        """
        return list(map(lambda x: self.__log_prob_sentence(x), sentence_list))


df_train = pd.read_csv("train.tsv", sep='\t')
df_test = pd.read_csv("task.tsv", sep='\t')

print(df_train.head(2))

print("Read ", df_train.shape, df_test.shape)

basic_lm = LanguageModel()

sentences_train = df_train["text"].tolist()
basic_lm.fit(sentences=sentences_train)

print("Trained")

test1, test2 = df_test["text1"], df_test["text2"]

logprob1, logprob2 = np.array(basic_lm.log_prob(test1)), np.array(basic_lm.log_prob(test2))

res = pd.DataFrame()
res["id"] = df_test["id"]
res["which"] = 0
res.loc[logprob2 >= logprob1, ["which"]] = 1

res.to_csv("submission.csv", sep=",", index=None, columns=["id", "which"])
