import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
import pickle

import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.utils.data import DataLoader

from lab_3.model import CharRNN, My_Dataset, BalancedBatchSampler, Config, EarlyStopping


if cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'


if __name__ == '__main__':
    config = Config(lr=2e-5, batch_size=16000, num_epochs=100)

    path_file = r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\features3.csv'
    df = pd.read_csv(path_file)

    int2char = dict(enumerate(set(df.grapheme)))
    char2int = {ch: ii for ii, ch in int2char.items()}

    int2phoneme = dict(enumerate(set(df.phoneme)))
    phoneme2int = {ch: ii for ii, ch in int2phoneme.items()}


    # готовим данные для загрузки в модель, определяющую фонемы по графемам
    df['int_char'] = [char2int[x] for x in df.grapheme]
    to_convert = ['subpart_of_speech', 'genesys', 'semantics1', 'semantics2', 'form', 'before_pause', 'after_pause',
                  'before_vowel', 'after_vowel', 'stressed_vowel', 'word', 'grapheme', 'phoneme', 'allophone', 'int_char']
    for col in to_convert:
        df[col] = df[col].astype('category')
    X = df.drop(columns=['word', 'phoneme', 'grapheme', 'allophone', 'stressed_vowel'])

    X = pd.get_dummies(X).values
    # X = pd.get_dummies(df.loc[:, ['int_char', 'stressed_vowel']]).values
    y = [phoneme2int[x] for x in df.phoneme]

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    batcher_train = DataLoader(My_Dataset(x_train, y_train), batch_size=config.batch_size,
                               sampler=BalancedBatchSampler(My_Dataset(x_train, y_train), y_train))
    batcher_val = DataLoader(My_Dataset(x_val, y_val), batch_size=config.batch_size, shuffle=False)

    model = CharRNN(X.shape[1], len(int2phoneme))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.001)
    early_stopping = EarlyStopping()

    model.to(device)
    min_loss = 1000

    #собственно тренируем модель
    for epoch in range(config.num_epochs):
        h = model.init_hidden(config.batch_size)
        model.train()
        correct = 0.
        total = 0.
        iter_loss = 0.
        for i, (items, classes) in enumerate(batcher_train):
            if classes.shape[0] != config.batch_size:
                break

            items = items.to(device)
            classes = classes.to(device)
            optimizer.zero_grad()

            h = tuple([each.data for each in h])
            outputs, h = model(items, h)

            loss = criterion(outputs, classes.long())
            iter_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data.long()).sum().item()
            total+=1

            torch.cuda.empty_cache()

        print('Iter {}, Train loss: {}, accuracy: {}'.format(epoch, np.round(iter_loss/total, 4), np.round(correct/total/config.batch_size, 2)))

        ############################
        # Validate
        ############################
        correct = 0.
        total = 0.
        iter_loss = 0.

        model.eval()  # Put the network into evaluate mode
        val_h = model.init_hidden(config.batch_size)

        for i, (items, classes) in enumerate(batcher_val):
            if classes.shape[0] != config.batch_size:
                break

            items = items.to(device)
            classes = classes.to(device)

            val_h = tuple([each.data for each in val_h])
            outputs, val_h = model(items, val_h)
            loss = criterion(outputs, classes.long())
            iter_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data.long()).sum().item()
            total += 1

        print('          Val loss: {}, accuracy: {}\n'.format(np.round(iter_loss/total, 4), np.round(correct/total/config.batch_size, 2)))

        early_stopping.update_loss(iter_loss)
        if early_stopping.stop_training():
            torch.save(model, r"C:\Users\denis\Documents\TTS_saved_model\net.pb")
            break







