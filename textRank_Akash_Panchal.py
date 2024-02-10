import re

import numpy as np
from nltk import sent_tokenize, word_tokenize

from nltk.cluster.util import cosine_distance

MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


def normalize_whitespace(text):
    """
    Translates multiple whitespace into single space character.
    If there is at least one new line character chunk is replaced
    by single LF (Unix new line) character.
    """
    return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text)


def _replace_whitespace(match):
    text = match.group()

    if "\n" in text or "\r" in text:
        return "\n"
    else:
        return " "


def is_blank(string):
    """
    Returns `True` if string contains only white-space characters
    or is empty. Otherwise `False` is returned.
    """
    return not string or string.isspace()


def get_symmetric_matrix(matrix):
    """
    Get Symmetric matrix
    :param matrix:
    :return: matrix
    """
    # print(f"matrix is: {matrix} it's dim {np.array(matrix).ndim} it's size {np.array(matrix).size} and len {len(matrix)}")
    # print(f"matrix.T = {matrix.T} it's dim {np.array(matrix.T).ndim} it's size {np.array(matrix.T).size} and len {len(matrix)}")
    # print(f"np.diag(matrix.diagonal()) is {np.diag(matrix.diagonal())}")
    # print(f"matrix + matrix.T - np.diag(matrix.diagonal()) is {matrix + matrix.T - np.diag(matrix.diagonal())}")
    return matrix + matrix.T - np.diag(matrix.diagonal())


def core_cosine_similarity(vector1, vector2):
    """
    measure cosine similarity between two vectors
    :param vector1:
    :param vector2:
    :return: 0 < cosine similarity value < 1
    """
    return 1 - cosine_distance(vector1, vector2)


'''
Note: This is not a summarization algorithm. This Algorithm pics top sentences irrespective of the order they appeared.
'''


class TextRank4Sentences():
    def __init__(self):
        self.damping = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 100  # iteration steps
        self.text_str = None
        self.sentences = None
        self.pr_vector = None

    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
        # print(f"sent1 = {sent1} len = {len(sent1)} \n\n sent2 = {sent2} len = {len(sent2)}\n")

        all_words = list(set(sent1 + sent2))
        # print(f"all_words = {all_words} len = {len(all_words)} \n")

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        # print(f"vector1 = {vector1} and vector2 = {vector2}")

        # build the vector for the first sentence
        for w in sent1:
            # print(f"w in sent1 = {w}")
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1
            # print(f"all_words.index(w) = {all_words.index(w)} w is {w}")
        # print(f"vector1 = {vector1}")
            
        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1
        #     print(f"all_words.index(w) = {all_words.index(w)} w is {w}")
        # print(f"vector2 = {vector2}")
        return core_cosine_similarity(vector1, vector2)

    def _build_similarity_matrix(self, sentences, stopwords=None):
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])
        # print(f"Empty similarity matrix {sm} and it's dim {sm.ndim} and it's len {len(sm)} and it's size {sm.size}\n")
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    # print(f"idx1 = {idx1} is equal to  idx2 ={idx2}")
                    continue

                sm[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2], stopwords=stopwords)
                # print(f"idx1 = {idx1} , idx2 ={idx2} and sm[idx1][idx2] is {sm[idx1][idx2]}")

        # Get Symmeric matrix
        sm = get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        # print(f"norm is {norm}")
        sm_norm = np.divide(sm, norm, where=norm != 0)  # this is ignore the 0 element in norm
        print(f"sm_norm is {sm_norm}")
        return sm_norm

    def _run_page_rank(self, similarity_matrix):

        pr_vector = np.array([1] * len(similarity_matrix))
        # print(f"pr_vector is {pr_vector}")
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            # print(f"np.matmul(similarity_matrix, pr_vector) is {np.matmul(similarity_matrix, pr_vector)}")
            # print(f"self.damping * np.matmul(similarity_matrix, pr_vector) is {self.damping * np.matmul(similarity_matrix, pr_vector)}")
            pr_vector = (1 - self.damping) + self.damping * np.matmul(similarity_matrix, pr_vector)
            print(f"pr_vector is {pr_vector}")
            # print(f"sum(pr_vector) is {sum(pr_vector)}")
            # print(f"abs(previous_pr - sum(pr_vector)) is {abs(previous_pr - sum(pr_vector))}")
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                # print("going to if")
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index):

        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def get_top_sentences(self, number=5):

        top_sentences = []

        if self.pr_vector is not None:
            # print(f"self.pr_vector is {self.pr_vector}")
            sorted_pr = np.argsort(self.pr_vector)
            print(f"sorted_pr is {sorted_pr}")
            sorted_pr = list(sorted_pr)
            # print(f"sorted_pr is {sorted_pr}")
            sorted_pr.reverse()
            print(f"sorted_pr.reverse() is {sorted_pr}")

            index = 0
            for epoch in range(number):
                print(f"sorted_pr[index] is {sorted_pr[index]}")
                sent = self.sentences[sorted_pr[index]]
                print(f"sent is {sent}")
                sent = normalize_whitespace(sent)
                top_sentences.append(sent)
                index += 1

        return top_sentences

    def analyze(self, text, stop_words=None):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

        tokenized_sentences = [word_tokenize(sent) for sent in self.sentences]

        similarity_matrix = self._build_similarity_matrix(tokenized_sentences, stop_words)

        self.pr_vector = self._run_page_rank(similarity_matrix)


text_str = '''
    The 1948 Winter Olympics, officially known as the V Olympic Winter Games (German: V. Olympische Winterspiele; French: Ves Jeux olympiques d'hiver; Italian: V Giochi olimpici invernali; Romansh: V Gieus olimpics d'enviern) and commonly known as St. Moritz 1948 (French: Saint-Moritz 1948; Romansh: San Murezzan 1948), were a winter multi-sport event held from 30 January to 8 February 1948 in St. Moritz, Switzerland. 
    The Games were the first to be celebrated after World War II; it had been twelve years since the last Winter Games in 1936.
    From the selection of a host city in a neutral country to the exclusion of Japan and Germany, the political atmosphere of the post-war world was inescapable during the 1948 Games. 
    The organizing committee faced several challenges due to the lack of financial and human resources consumed by the war. 
    These were the first of two winter Olympic Games under the IOC presidency of Sigfrid Edström.
    There were 28 nations that marched in the opening ceremonies on 30 January 1948.
    Bibi Torriani played for the Switzerland men's national ice hockey team, and became the first ice hockey player to recite the Olympic Oath on behalf of all athletes.
    Nearly 670 athletes competed in 22 events in four sports.
    The 1948 Games also featured two demonstration sports: military patrol, which later became the biathlon, and winter pentathlon, which was discontinued after these Games. 
    Notable performances were turned in by figure skaters Dick Button and Barbara Ann Scott and skier Henri Oreiller. 
    Most of the athletic venues were already in existence from the first time St. Moritz hosted the Winter Games in 1928. 
    All of the venues were outdoors, which meant the Games were heavily dependent on favorable weather conditions.
    '''

tr4sh = TextRank4Sentences()
tr4sh.analyze(text_str)
print(tr4sh.get_top_sentences(5))