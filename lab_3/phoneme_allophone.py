# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score

import torch
import torch.cuda as cuda
from torch.utils.data import DataLoader

from lab_3.model import CharRNN, Config, My_Dataset
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

def model_x(model, x_test, y):
    preds = model.predict(x_test)
    score1 = f1_score(y, preds, average='micro')
    score2 = f1_score(y, preds, average='macro')
    accuracy = balanced_accuracy_score(y, preds)
    # roc_auc = roc_auc_score(y, preds)
    print("Test metrics: [f1 score: micro - %0.3f, macro - %0.3f], \n"
          " [accuracy: %0.3f])" % (score1, score2, accuracy))
    return preds, score1, score2

if __name__ == '__main__':
    #взял другие данные для теста
    path_file = r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\features_for_test.csv'
    df = pd.read_csv(path_file)
    df = df.iloc[:int(df.shape[0]/10), :]


    int2char = dict(enumerate(set(df.grapheme)))
    char2int = {ch: ii for ii, ch in int2char.items()}

    int2phoneme = dict(enumerate(set(df.phoneme)))
    phoneme2int = {ch: ii for ii, ch in int2phoneme.items()}

    int2allophone = dict(enumerate(set(df.allophone)))
    allophone2int = {ch: ii for ii, ch in int2allophone.items()}

    df['int_char'] = [char2int[x] for x in df.grapheme]
    df['int_phoneme'] = [phoneme2int[x] for x in df.phoneme]
    to_convert = ['subpart_of_speech', 'genesys', 'semantics1', 'semantics2', 'form', 'before_pause', 'after_pause',
                  'before_vowel', 'after_vowel', 'stressed_vowel', 'word', 'grapheme', 'phoneme', 'allophone',
                  'int_phoneme', 'int_char']
    for col in to_convert:
        df[col] = df[col].astype('category')

    X_1 = df.drop(columns=['word', 'phoneme', 'grapheme', 'allophone', 'stressed_vowel', 'int_phoneme'])
    X_1['temp'] = np.zeros((X_1.shape[0],))
    X_1 = pd.get_dummies(X_1).values

    y_1 = [phoneme2int[x] for x in df.phoneme]

    x_1_train, x_1_val, y_1_train, y_1_val = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

    X_2 = df.drop(columns=['word', 'phoneme', 'grapheme', 'allophone', 'int_char'])
    # X_2 = pd.get_dummies(X_2).values
    y_2 = [allophone2int[x] for x in df.allophone]

    x_2_train, x_2_val, y_2_train, y_2_val = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

    #обучаем модель фонема-аллофон случайным лесом
    clf = RandomForestClassifier(max_depth=50,
                n_estimators=200, random_state=0)
    t = pd.get_dummies(x_2_train).values
    clf.fit(pd.get_dummies(x_2_train).values, y_2_train)

    #теперь посмотрим как отработает полный пайплайн на валидационной части


    #загружаем предобученную модель графема-фонема
    if cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    net = torch.load(r"C:\Users\denis\Documents\TTS_saved_model\net.pb")
    net.to(device)
    net.eval()

    config = Config(lr=2e-5, batch_size=16000, num_epochs=100)
    batcher_test = DataLoader(My_Dataset(x_1_val, y_1_val), batch_size=config.batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    torch.manual_seed(7)
    h = net.init_hidden(config.batch_size)

    predicted_phonemes = []

    for i, (items, classes) in enumerate(batcher_test):
        if classes.shape[0] != config.batch_size:
            break
        items = items.to(device)
        classes = classes.to(device)
        h = tuple([each.data for each in h])

        outputs, h = net(items, h)

        _, predicted = torch.max(outputs.data, 1)

        predicted_phonemes += list(predicted.cpu().numpy())

    x_2_val, y_2_val = x_2_val.iloc[:64000,:], y_2_val[:64000]
    x_2_val['int_phoneme'] = predicted_phonemes
    x_2_val['int_phoneme'].astype('category')

    preds, score1, score2 = model_x(clf, pd.get_dummies(x_2_val).values, y_2_val)








