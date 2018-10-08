"""
    Homework #2

    The goal is to fix the sentences from broken.csv; the quality measure is the exact match of fixed sentences and groundtruths.

    https://www.kaggle.com/c/nru-itmo-string-metrics-lab

    This source code is not necessary to use.

    You are not allowed to use spell checkers, otherwise your will be not accepted.

    The usage of opensource libraries with the edit distances (or even more language modeling) is welcome an is also implied.

    If you are not sure what tool to use, please, ask you teachers.

    When the quality measure of your solution exceeds the baseline or the homework deadline is near, 
    please, prepare your code and submit it to kaggle site.

    THe private results can be ask from teachers.

    P.S.: In real life, it is more useful to use logs, not debug printing
    P.P.S.: It is better not to delay the task on the last day, because it is calculated task is not very fast.

"""
import codecs
import csv
import time
from collections import Counter
from collections import defaultdict
from functools import lru_cache

import stringdist as Distancer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class Speller(object):
    """
        Search of most similar words by the number of common n-grams 
        and sotring them by distance measure
    """

    def __init__(self, n_candidates_search=20):
        """
        :param n_candidates_search: the number of candidates string in search
        """
        # todo: maybe it is an important parametr?
        self.n_candidates = n_candidates_search

    def fit(self, words_list):
        """
            Speller fitting
        """

        checkpoint = time.time()
        self.words_list = words_list

        # todo: maybe the n-gramms size is matter?
        # todo: maybe, it is better to work with non-binary values?
        self.vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(3, 3), binary=False)
        encoded_words = self.vectorizer.fit_transform(words_list).tocoo()
        self.index = defaultdict(set)

        # build the dictionatry that maps ngramms to a set of terms
        for i in zip(encoded_words.row, encoded_words.col):
            self.index[i[1]].add(i[0])

        print("Speller fitted in", time.time() - checkpoint)
        return self

    @lru_cache(maxsize=1000000)
    def rectify(self, word):
        """
            Speller predictions
        """

        # the query that is mapped to ngrams
        char_ngrams_list = self.vectorizer.transform([word]).tocoo().col
      #  print('--------')
      #  print(word)
        # calculate the number of matches for each term
        counter = Counter()

        for token_id in char_ngrams_list:
            for word_id in self.index[token_id]:
                counter[word_id] += 1

        # search for the nearest term from the selected terms
        closest_word = word
        minimal_distance = 1000
        # search for "good" fix from the top of matches by n-gramms
        for suggest in counter.most_common(n=self.n_candidates):

            suggest_word = self.words_list[suggest[0]]
            # TODO: your code here
            # you can use any libraries and sources except the original texts

            distance = Distancer.levenshtein(word, suggest_word)

            if distance < minimal_distance:
                minimal_distance = distance
                closest_word = suggest_word
             #   print(closest_word)
        return closest_word


if __name__ == "__main__":

    np.random.seed(0)

    # read the dictionary of correct words
    words_set = set(line.strip() for line in codecs.open("words2.txt", "r", encoding="utf-8"))
    words_list = sorted(list(words_set))

    # сreate speller
    speller = Speller()
    speller.fit(words_list)

    # read the sentences
    df = pd.read_csv("broken.csv").head(50)
    # todo: change the line to comment line to run the code on the full dataset
 # df = pd.read_csv("broken.csv")


    checkpoint1 = time.time()
    total_rectification_time = 0.0
    total_sentences_rectifications = 0.0

    y_submission = []
    counts = 0

    # fix while collecting the counters and track the time
    for i in range(df.shape[0]):

        counts += 1

        if counts % 100 == 0:
            print("Rows processed", counts)

        start = time.time()
        mispelled_text = df["text"][i]
        mispelled_tokens = mispelled_text.split()

        print(mispelled_text)

        was_rectified = False

        for j in range(len(mispelled_tokens)):
            if mispelled_tokens[j] not in words_set:
                rectified_token = speller.rectify(mispelled_tokens[j])
                mispelled_tokens[j] = rectified_token
                was_rectified = True

        if was_rectified:
            mispelled_text = " ".join(mispelled_tokens)
            total_rectification_time += time.time() - start
            total_sentences_rectifications += 1.0

        y_submission.append(mispelled_text)

    checkpoint2 = time.time()

    print("elapsed", checkpoint2 - checkpoint1)
    print("average speller time", total_rectification_time / float(total_sentences_rectifications))

    submission = pd.DataFrame({"id": df["id"], "text": y_submission}, columns=["id", "text"])
    submission.to_csv("baseline_submission.csv", index=None, encoding="utf-8", quotechar='"',
                      quoting=csv.QUOTE_NONNUMERIC)
