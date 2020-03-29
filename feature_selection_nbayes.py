import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction import text
import csv


# methods
##########

# load Excel spreadsheet containing the data that will be used
def prepare_data():
    # this will be implemented on my office client
    return None

# get keys from 
def getKeys(vocabulary, index_values, probabilities):
    keys = [{word:probabilities[index]} for word, index in vocabulary.items() if index in index_values]
    return keys

def getNTopWords(n, probabilities, dict_vocabulary):
    top_n_idx = np.argsort(probabilities)[-n:]
    top_n_values = getKeys(dict_vocabulary, top_n_idx, probabilities)
    return top_n_values

def getTfidfValues(words):
    vectorizer = TfidfVectorizer(stop_words = 'english')
    tfidf_matrix =  vectorizer.fit_transform(words)
    feature_names = vectorizer.get_feature_names()
    doc = 0
    feature_index = tfidf_matrix[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
    
    #print
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        print(w, s)
    return None
    
def visualizeModel(categories,clf):
    # get test data
    newsgroup_test = fetch_20newsgroups(subset='test', categories=categories)
    # transform test data
    vectors_test = vectorizer.transform(newsgroup_test.data)
    
    # predictions and confusion matrix
    predictions = clf.predict(vectors_test)

    print(metrics.confusion_matrix(newsgroup_test.target, predictions))
    print(metrics.classification_report(newsgroup_test.target, predictions))
    
    # below is how we did it in the course, however in my data set
    # there are about 400 classes, therefore this cannot be used
    """
    metrics.plot_confusion_matrix(clf, vectors_test, newsgroup_test.target, normalize='pred')
    plt.xticks(np.arange(0, 6), sorted(categories), rotation=20)
    plt.yticks(np.arange(0, 6), sorted(categories))
    plt.show()
    """
    return None

def createTopNWordsDict(n, clf):
    # create top n words
    # loop through probability table
    n = 20
    i = 0
    result = {}
    classes = clf.classes_
    classes_names = sorted(categories)
    for row in probabilities:
        result.update({ classes_names[i] : getNTopWords(n, row, dict_vocabulary)})
        i += 1
    return result
    
def writeFeaturesCSV(dict):
    w = csv.writer(open("features.csv", "w"))
    for key, val in dict.items():
        for feature in val:
            w.writerow([key,list(feature.keys())[0]])

# read data in
################

# read in data - here I took the same categories as we had in the lesson
categories = ['soc.religion.christian', 'comp.graphics', 'sci.med', 'comp.windows.x', 'sci.space', 'alt.atheism']
newsgroup_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)


# define stop_words
####################
# additional stopwords that I evaluated with the feature results and positive class
my_additional_stop_words = ['writes', 'think', 'subject','know','say','using','thanks','like','just','people','does','posting','don','com','organization','lines','edu', 'article','reply','looking']
# union with pre-defined stopwords from the package
stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)


# calculate tfidf
##################

# vectorize
# TFIDF performs better - did also try CountVectorizer
stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)
print("the following stop words are used:")
print(stop_words)
vectorizer = TfidfVectorizer(stop_words = stop_words)

vectors_train = vectorizer.fit_transform(newsgroup_train.data)

# get full vocabulary 
dict_vocabulary = vectorizer.vocabulary_

# overview of tf-idf values
feature_names = vectorizer.get_feature_names()
dense = vectors_train.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
#print(df)
print("-----------------------------------------------------------")

# overview of TFIDF values
###########################

# was only to check, but not really used
#getTfidfValues(newsgroup_train.data)

# multinomial naive bayes
clf = MultinomialNB(alpha=0.1)
clf.fit(vectors_train, newsgroup_train.target)


# this is to check, if removal of stopwords hurts the model
# visualize performance of model
############################################################
visualizeModel(categories,clf)

# get classes and print them
classes = clf.classes_
classes_names = sorted(categories)
print("Data set contains the following classes:")
print(classes)
print(sorted(categories))
print("-----------------------------------------------------------")

# feature log probability
probabilities = np.exp(clf.feature_log_prob_)
print("")
print("These are the full feature probabilities for each class")
print(probabilities.shape)
print(probabilities)
print("-----------------------------------------------------------")

# get top 20 words
result = createTopNWordsDict(20, clf)
# print dictionary
for k, v in result.items():
    print(k, ' : ',v)
    print("----------------")
    
print("-----------------------------------------------------------")
print("")

# save features in text document to upload them to the target system
writeFeaturesCSV(result)


######################################################################################
# this is another approach I tried to do, mostly to improve my stopwords
# top 20 for positive and negative class
######################################################################################

# For positive class
sorted_prob_class_1_ind = clf.feature_log_prob_[1, :].argsort()
# For negative class
sorted_prob_class_0_ind = clf.feature_log_prob_[0, :].argsort()

features_lst = vectorizer.get_feature_names()

Most_imp_words_1 = []
Most_imp_words_0 = []

for index in sorted_prob_class_1_ind[-20:-1]:
    Most_imp_words_1.append(features_lst[index])

for index in sorted_prob_class_0_ind[-20:-1]:
    Most_imp_words_0.append(features_lst[index])

print("20 most imp features for positive class:\n")
print(Most_imp_words_1)

print("\n" + "-"*100)

print("\n20 most imp features for negative class:\n")
print(Most_imp_words_0)