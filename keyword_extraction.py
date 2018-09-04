'''
    Name: Sumanth Doddapaneni
    Roll: 20160020125
    Last Update:
    Work: "Key term extarction from a single doc"
'''

# importing req libraries
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import nltk
import numpy as np
import itertools
from scipy.sparse import csr_matrix, lil_matrix

# creating stop list
#nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
# SMART stop words - 571
file = "stopwords.txt"
stoplist = []
with open(file) as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    for line in lines:
        stoplist.append(line)
stopwords.extend(stoplist)

#print(stopwords)

# lemantizing to remove multiple occurances of similar words
lemmatiser = WordNetLemmatizer()
stemmer = PorterStemmer()

file = open("turing.txt")

# def structure(word):
#     if word[-1] == '.':
#         word = word[:-1]
#     elif word[-1] == ',':
#         word = word[:-1]
#     elif word[-1] == 's':
#         word = word[:-1]
#     elif word[-1] == '"':
#         word = word[:-1]
#     elif word[-1] == "'":
#         word = word[:-1]
#     elif word[-1] == ")":
#         word = word[:-1]
#     elif word[-1] == "?":
#         word = word[:-1]
#     elif word[0] == "(":
#         word = word[1:]
#     elif word[0] == ',':
#         word = word[1:]
#     elif word[0] == 's':
#         word = word[1:]
#     elif word[0] == '"':
#         word = word[1:]
#     elif word[0] == "'":
#         word = word[1:]
#
#     return word

# creating dictionary for the frequencies of words
word_freq = {}
for word in file.read().split():
    word = word.lower()
    if word not in stopwords:
        #word = structure(word)
        word = lemmatiser.lemmatize(word, pos="v")
        #word = stemmer.stem(word)
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] += 1

sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
# print(sorted_word_freq)

# top 10 frequent words list
freq_words = []
for k, v in sorted_word_freq:
    if len(freq_words) <= 10:
        freq_words.append(k)
print(freq_words)

# list of all the sentences
content = []
filename = 'turing.txt'
with open(filename) as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    words = []
    for line in lines:
        line = line.lower()
        words = line.split()
        content.append(words)
        words = []
# print(content)

words = word_freq.keys()
# creating co-occurance matrix
'''def create_cooc_matrix(freq_words, documents):
    word_to_id = dict(zip(freq_words, range(len(freq_words))))
    documents_as_ids = [np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype('uint32') for doc in documents]
    row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
    data = np.ones(len(row_ind), dtype='uint32')
    max_word_id = max(itertools.chain(*documents_as_ids)) + 1
    docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id))
    words_cooc_matrix = docs_words_matrix.T * docs_words_matrix
    words_cooc_matrix.setdiag(0)
    return words_cooc_matrix, word_to_id

words_cooc_matrix, word_to_id = create_cooc_matrix(words, content)
print(words_cooc_matrix.all())
words_cooc_matrix = np.asarray(words_cooc_matrix)
print(words_cooc_matrix)'''

# initialising empty dictionary
cooc_matrix = {}
for key in word_freq:
    cooc_matrix[key] = {}

# creating dictionary
def create_cooc_matrix():
    for key in word_freq:
        with open(filename) as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
            words = []
            for line in lines:
                line = line.lower()
                words = line.split()
                if key in words:
                    words.remove(key)
                    for word in words:
                        #word = structure(word)
                        if word in cooc_matrix[key].keys():
                            cooc_matrix[key][word] += 1
                        else:
                            cooc_matrix[key][word] = 1
create_cooc_matrix()
#print(cooc_matrix['machine'])

X2_dict = {}
def X2_test():
    for key in word_freq:
        X2 = 0
        nw = sum(cooc_matrix[key].values())
        if nw < 2:
            continue
        for word in freq_words:
            pg = sum(cooc_matrix[word].values()) / len(word_freq)
            try:
                freq_w_g = cooc_matrix[key][word]
            except:
                freq_w_g = 0
            X2 += (freq_w_g - (nw * pg))**2 / (nw * pg)
            X2_dict[key] = X2

X2_test()
sorted_dict = sorted(X2_dict.items(), key=lambda x: x[1], reverse=True)
print(sorted_dict)
