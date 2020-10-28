# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from navec import Navec
from gensim.models import fasttext
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from nltk import word_tokenize
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def anova(model):
    anova_filter = SelectKBest(f_regression, k=19)
    anova_svm = make_pipeline(anova_filter, model)
    return anova_svm

def svm(kernel='rbf', C=10, gamma='scale'):
    svc = SVC(kernel=kernel, class_weight='balanced', C=C, degree=5, tol=1e-5, gamma=gamma)
    return svc

def linear_svm():
    svc = LinearSVC(C = 1, random_state=0, tol=1e-5, dual=False, fit_intercept=True, class_weight='balanced')
    return svc

def voting(models):
    l = []
    for i, m in enumerate(models):
        l.append(("clf_{}".format(i), m))
    eclf = VotingClassifier(estimators = l,
                            voting = 'hard')
    return eclf

def adaboost():
    adaboost = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=50),
                n_estimators=200, random_state=0)
    return adaboost

def train(model, data, y):
    kf = KFold(n_splits=2, random_state=None)
    scores = []
    for train_index, val_index in kf.split(data):
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = y[train_index], y[val_index]
        scores.append(model.fit(X_train, y_train).score(X_val, y_val))
    scores = np.array(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return model, scores

def test(model, x_test, y):
    preds = model.predict(x_test)
    score1 = f1_score(y, preds, average='micro')
    score2 = f1_score(y, preds, average='macro')
    accuracy = balanced_accuracy_score(y, preds)
    roc_auc = roc_auc_score(y, preds)
    print("Test metrics: [f1 score: micro - %0.3f, macro - %0.3f], \n"
          " [accuracy: %0.3f], [roc_auc: %0.3f])" % (score1, score2, accuracy, roc_auc))
    return preds, score1, score2

def get_pre_learned():
    i = 0
    embs = []
    words = []
    with open(r'C:\Users\denis\Documents\180\model.txt', 'r', encoding='utf-8') as f:
        text = f.readlines()
        for row in text:
            data = row.split(' ')
            word = data[0].split('_')[0]

            emb = [float(i) for i in data[1:]]
            words.append(word)
            embs.append(emb)
    return words, embs


def get_fasttext(df, pre_words, pre_embs):
    embs = []

    words = np.array(df['word'].values).reshape(df.shape[0], 1)

    model = fasttext.FastText(words, size=100, window=2, min_count=0)
    for word in tqdm(df["word"].values, total=df.shape[0]):
        if word in pre_words:
            i = pre_words.index(word)
            emb = pre_embs[i]
        else:
            emb = model.wv[word]
        embs.append(emb)
    embs = np.array(embs)
    np.save('./fasttext.npy', embs)
    return embs


if __name__ == '__main__':
    corpus = pd.read_csv('./data/corpus.csv')
    vectors = pd.read_csv('./data/vectors.csv')
    y = corpus['ispunctuation'].values
    features = np.load("./features.npy", allow_pickle=True)
    feats = np.load('./word2vec.npy')
    rms = np.load('./rms.npy')
    X_train, X_test, y_train, y_test = train_test_split(rms, y, test_size=0.1, random_state=42)
    # model = svm(kernel='rbf')
    # model2 = svm(kernel='rbf', C=1, gamma='auto')
    # model3 = svm(kernel='poly')
    # model4 = voting([model, model2, model3])
    adaboost_m = adaboost()
    model, scores = train(adaboost_m, X_train, y_train)
    preds, score1, score2 = test(adaboost_m, X_test, y_test)
    print(adaboost_m.get_params())
    #get_word2vec(vectors)
