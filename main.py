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

def rf():
    classifier = RandomForestClassifier(max_depth=50,
                n_estimators=200, random_state=0)
    return classifier

def train(model, data, y):
    kf = KFold(n_splits=3, random_state=None)
    scores = []
    for train_index, val_index in kf.split(data):
        X_train, X_val = data.iloc[train_index,:].values, data.iloc[val_index,:].values
        y_train, y_val = np.array(y)[train_index], np.array(y)[val_index]
        model.fit(X_train, y_train)
        scores.append(model.fit(X_train, y_train).score(X_val, y_val))
    scores = np.array(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return model, scores

def model_x(model, x_test, y):
    preds = model.predict(x_test)
    score1 = f1_score(y, preds, average='micro')
    score2 = f1_score(y, preds, average='macro')
    accuracy = balanced_accuracy_score(y, preds)
    # roc_auc = roc_auc_score(y, preds)
    # print("Test metrics: [f1 score: micro - %0.3f, macro - %0.3f], \n"
    #       " [accuracy: %0.3f], [roc_auc: %0.3f])" % (score1, score2, accuracy, roc_auc))
    return preds, score1, score2


if __name__ == '__main__':
    features_path = r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\features.csv'
    df = pd.read_csv(features_path)
    df = df.fillna(-1)

    # FIRST TASK
    y = df['pause'] != -1
    y1 = [int(x) for x in y]
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['intonation', 'pause', 'vowel', 'reduct', 'word']), y1, test_size=0.1, random_state=42)

    rf = rf()
    model, scores = train(rf, X_train, y_train)
    preds, score1, score2 = model_x(model, X_test, y_test)
    print("Binary classificator results: accuracy: {}, f1-macro: {}, f1-micro: {}".format(np.mean(scores), score1, score2))

    # SECOND TASK
    dict_target = {-1: -1, 'weak': 0, 'long': 1, 'x-long': 2, 'spelling': 3}
    y2 = df.pause.map(lambda x: dict_target[x])
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['intonation', 'pause', 'vowel', 'reduct', 'word']), y2, test_size=0.1, random_state=42)

    model, scores = train(rf, X_train, y_train)
    preds, score1, score2 = model_x(model, X_test, y_test)
    print("Multi-class classificator results: accuracy: {}, f1-macro: {}, f1-micro: {}".format(np.mean(scores), score1,
                                                                                          score2))

    # THIRD TASK
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['intonation', 'pause', 'vowel', 'reduct', 'word']), df.intonation, test_size=0.1, random_state=42)

    model, scores = train(rf, X_train, y_train)
    preds, score1, score2 = model_x(model, X_test, y_test)
    print("intonation classificaton results: accuracy: {}, f1-macro: {}, f1-micro: {}".format(np.mean(scores), score1,
                                                                                          score2))

    print(model.get_params())






