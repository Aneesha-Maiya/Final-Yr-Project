import re
import nltk

import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nltk.cluster.util import cosine_distance
df = pd.read_csv('./datasets/news_summary(50-200).csv',encoding = "unicode_escape")

MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)
# nltk.download('punkt')

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
        print(f"\n pr_vector is {pr_vector}")
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            print(f"\n np.matmul(similarity_matrix, pr_vector) is {np.matmul(similarity_matrix, pr_vector)}")
            print(f"self.damping * np.matmul(similarity_matrix, pr_vector) is {self.damping * np.matmul(similarity_matrix, pr_vector)}")
            pr_vector = (1 - self.damping) + self.damping * np.matmul(similarity_matrix, pr_vector)
            print(f"pr_vector is {pr_vector}")
            print(f"sum(pr_vector) is {sum(pr_vector)}")
            print(f"abs(previous_pr - sum(pr_vector)) is {abs(previous_pr - sum(pr_vector))}")
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                print("going to if")
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index):

        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def get_top_sentences(self, number):

        top_sentences = []

        if self.pr_vector is not None:
            print(f"\n self.pr_vector is {self.pr_vector}")
            sorted_pr = np.argsort(self.pr_vector)
            print(f"sorted_pr is {sorted_pr}")
            sorted_pr = list(sorted_pr)
            # print(f"sorted_pr is {sorted_pr}")
            sorted_pr.reverse()
            print(f"sorted_pr.reverse() is {sorted_pr}")

            sorted_pr = sorted_pr[0:number]
            sorted_pr = sorted(sorted_pr)
            index = 0
            for epoch in range(number):
                print(f"sorted_pr[index] is {sorted_pr[index]}")
                sent = self.sentences[sorted_pr[index]]
                print(f"sent is {sent}")
                sent = normalize_whitespace(sent)
                top_sentences.append(sent)
                index += 1

        return top_sentences,sorted_pr

    def analyze(self, text, stop_words=None):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

        tokenized_sentences = [word_tokenize(sent) for sent in self.sentences]

        similarity_matrix = self._build_similarity_matrix(tokenized_sentences, stop_words)
        # print(f"Simialirity matrix: \n {similarity_matrix}")

        self.pr_vector = self._run_page_rank(similarity_matrix)
        # print(f"self.pr_vector = {self.pr_vector}")

# text_str = '''
# The black-necked grebe or eared grebe (Podiceps nigricollis) is a member of the grebe family of water birds. 
# It was described in 1831 by Christian Ludwig Brehm. There are currently three accepted subspecies, including the nominate subspecies. 
# Its breeding plumage features distinctive ochre-coloured feathers which extend behind its eye and over its ear coverts. 
# The rest of the upper parts, including the head, neck, and breast, are coloured black to blackish brown. 
# The flanks are tawny rufous to maroon-chestnut, and the abdomen is white. 
# In its non-breeding plumage, this bird has greyish-black upper parts, including the top of the head and a vertical stripe on the back of the neck. 
# The flanks are also greyish-black. 
# The rest of the body is a white or whitish colour. 
# The juvenile has more brown in its darker areas. 
# The subspecies californicus can be distinguished from the nominate by the former's usually longer bill. 
# The other subspecies, P. n. gurneyi, can be differentiated by its greyer head and upper parts and by its smaller size. 
# P. n. gurneyi can also be told apart by its lack of a non-breeding plumage. 
# This species is present in parts of Africa, Eurasia, and the Americas.
# The black-necked grebe uses multiple foraging techniques. 
# Insects, which make up the majority of this bird's diet, are caught either on the surface of the water or when they are in flight. 
# It occasionally practices foliage gleaning. 
# This grebe dives to catch crustaceans, molluscs, tadpoles, and small frogs and fish. 
# When moulting at saline lakes, this bird feeds mostly on brine shrimp. 
# The black-necked grebe makes a floating cup nest on an open lake. 
# The nest cup is covered with a disc. 
# This nest is located both in colonies and by itself. 
# During the breeding season, which varies depending on location, this species will lay one (sometimes two) clutch of three to four eggs. 
# The number of eggs is sometimes larger due to conspecific brood parasitism. 
# After a 21-day incubation period, the eggs hatch, and then the nest is deserted. 
# After about 10 days, the parents split up the chicks between themselves. 
# After this, the chicks become independent in about 10 days, and fledge in about three weeks.
# Although it generally avoids flight, the black-necked grebe travels as far as 6,000 kilometres (3,700 mi) during migration. 
# In addition, it becomes flightless for at least a month after completing a migration to reach an area where it can safely moult. 
# During this moult, the grebe can double in weight. 
# The migrations to reach these areas are dangerous, sometimes with thousands of grebe deaths. 
# In spite of this, it is classified as a least concern species by the International Union for Conservation of Nature (IUCN). 
# It is likely that this is the most numerous grebe in the world. 
# There are potential threats to it, such as oil spills, but these are not likely to present a major risk to the overall population.
# The black-necked grebe usually measures between 28 and 34 centimetres (11 and 13 in) in length and weighs 265 to 450 grams (9.3 to 15.9 oz). 
# The bird has a wingspan range of 20.5-21.6 in (52-55 cm).
# The nominate subspecies in breeding plumage has the head, neck, breast, and upper parts coloured black to blackish brown, with the exception of the ochre-coloured fan of feathers extending behind the eye over the eye-coverts and sides of the nape. 
# This eye is mostly red, with a narrow and paler yellow ring on the inner parts of the eye and an orange-yellow to pinkish-red orbital ring.
# The thin, upturned bill,on the other hand, is black, and is connected to the eye by a blackish line starting at the gape. 
# Sometimes, the foreneck can be found to be mostly tinged brown. 
# The upperwing is blackish to drab brown in colour and has a white patch formed by the secondaries and part of the inner primaries. 
# The flanks are coloured tawny rufous to maroon-chestnut and have the occasional blackish fleck. 
# The underwing and abdomen is white, with an exception to the former being the dark tertials and the mostly pale grey-brown outer primaries. 
# The legs are a dark greenish grey. The sexes are similar.
# In non-breeding plumage, the nominate has greyish-black upper parts, cap, nape, and hindneck, with the colour on the upper portion of the latter being contained in a vertical stripe. 
# The dark colour of the cap reaches below the eye and can be seen, diffused, to the ear-coverts. 
# Behind the ear-coverts on the sides of the neck, there are white ovals. 
# The rest of the neck is grey to brownish-grey in colour and has white that varies in amount. 
# The breast is white, and the abdomen is whitish. 
# The flanks are coloured in a mix of blackish-grey with white flecks. 
# The colour of the bill when not breeding differs from that of the breeding plumage, with the former being significantly more grey.
#     '''

text_str = (df.loc[2]['ctext'])

tr4sh = TextRank4Sentences()
tr4sh.analyze(text_str)
summary,sorted_sent_index = tr4sh.get_top_sentences(3)
print(summary)
print(sorted_sent_index)
# print((df.loc[2]['ctext']))
print(text_str)

# Convert the matrix into a NetworkX graph
G = nx.Graph()
thickness_levels = {
    (0.0, 0.1): 1,  # thickness for weights in the range (0.0, 0.1)
    (0.1, 0.2): 2,  # thickness for weights in the range (0.1, 0.2)
    (0.2, 0.3): 4,  # thickness for weights in the range (0.2, 0.3)
    (0.3, 0.4): 6   # thickness for weights in the range (0.3, 0.4)
}
sentences = sent_tokenize(text_str)
tokenized_sentences = [word_tokenize(sent) for sent in sentences]
matrix = tr4sh._build_similarity_matrix(tokenized_sentences)
sum_matrix = []
m,n = np.shape(matrix)
print(m,n)
# Add nodes to the graph
num_nodes = len(matrix)
G.add_nodes_from(range(num_nodes))
for i in range(0,m):
    sum_matrix.append(np.sum(matrix[i]))
print(sum_matrix)
# print(f"matrix = {matrix}")
# Add edges to the graph with their weights
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        weight = matrix[i][j]
        if weight != 0:
            G.add_edge(i, j, weight=weight)

# Draw the graph
final_pr_vector = tr4sh.pr_vector
node_sizes = []
node_colors = []
node_info = {}
for i in range(0,num_nodes):
    node_sizes.append(500*final_pr_vector[i])
for i in range(0,num_nodes):
    if (i in sorted_sent_index):
        node_colors.append('#1E90FF')
    else:
        node_colors.append('#00FF7F')
for i in range(0,num_nodes):
    node_info[i] = f"S{i} = {round(final_pr_vector[i],2)}"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),gridspec_kw={'width_ratios': [7, 3]})
pos = nx.shell_layout(G)  # positions for all nodes
for (start, end, weight) in G.edges(data='weight'):
    thickness = 1  # default thickness
    for (low, high), level in thickness_levels.items():
        if low < weight <= high:
            thickness = level
            break
nx.draw_networkx_edges(G, pos, edgelist=[(start, end)], width=thickness, ax = ax1)
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=10, edge_color = '#262626', 
edgecolors = 'black',font_color = 'black', ax = ax1)
# nx.draw_networkx_labels(G, pos, labels=node_info, font_color='red', font_size = 10)
edge_labels = {(i, j): round(d['weight'], 2) for i, j, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red',font_size = 8)

print(G.nodes())
table_data = [[node_info[node],node_info[node+1]] for node in range(0,len(sorted(G.nodes())),2)]
print("Table data ",table_data)
table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(0.6, 1.0)

ax1.axis('off')
ax1.set_title("Networkx Graph Plot")
legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) 
for color, label in zip(['#1E90FF', '#00FF7F'], ['Special Nodes', 'Other Nodes'])]
ax1.legend(handles=legend_handles, loc='lower right')
ax2.axis('off')
ax2.set_title("Node_Scores")
plt.show()
