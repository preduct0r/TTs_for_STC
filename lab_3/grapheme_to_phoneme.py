import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score

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
    config = Config(lr=0.000001, batch_size=16384, num_epochs=100)

    path_file = r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\features3.csv'
    df = pd.read_csv(path_file)
    df = df.fillna(-1)
    to_convert = ['subpart_of_speech', 'genesys', 'semantics1', 'semantics2', 'form', 'before_pause', 'after_pause',
                  'before_vowel', 'after_vowel']
    for col in to_convert:
        df[col] = df[col].astype('category')


    int2char = dict(enumerate(set(df.grapheme)))
    char2int = {ch: ii for ii, ch in int2char.items()}

    int2phoneme = dict(enumerate(set(df.phoneme)))
    phoneme2int = {ch: ii for ii, ch in int2phoneme.items()}

    df['int_char'] = [char2int[x] for x in df.grapheme]

    X = pd.get_dummies(df.drop(columns=['word', 'grapheme', 'phoneme', 'allophone'])).values
    y = [phoneme2int[x] for x in df.phoneme]

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    batcher_train = DataLoader(My_Dataset(x_train, y_train), batch_size=config.batch_size,
                               sampler=BalancedBatchSampler(My_Dataset(x_train, y_train), y_train))
    batcher_val = DataLoader(My_Dataset(x_val, y_val), batch_size=config.batch_size, shuffle=False)

    model = CharRNN(X.shape[1], len(int2phoneme))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    early_stopping = EarlyStopping()

    model.to(device)
    min_loss = 1000

    for epoch in range(config.num_epochs):
        h = model.init_hidden(config.batch_size)
        model.train()
        for i, (items, classes) in enumerate(batcher_train):
            if classes.shape[0] != config.batch_size:
                break

            items = items.to(device)
            classes = classes.to(device)
            optimizer.zero_grad()

            h = tuple([each.data for each in h])
            outputs, h = model(items, h)

            loss = criterion(outputs, classes.long())
            # iter_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            # correct += (predicted == classes.data.long()).sum()
            #
            # f_scores += f1_score(predicted.cpu().numpy(), classes.data.cpu().numpy(), average='macro')
            #
            # iterations += 1

            torch.cuda.empty_cache()

        # train_loss.append(iter_loss / iterations)
        # train_fscore.append(f_scores / iterations)

        ############################
        # Validate
        ############################
        iter_loss = 0.0
        # correct = 0
        # f_scores = 0
        # iterations = 0

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
            iter_loss += loss.item()/len(batcher_val)

            # _, predicted = torch.max(outputs.data, 1)
            # correct += (predicted == classes.data.long()).sum()
            #
            # f_scores += f1_score(predicted.cpu().numpy(), classes.data.cpu().numpy(), average='macro')
            #
            # iterations += 1

        # valid_loss.append(iter_loss / iterations)
        # valid_fscore.append(f_scores / iterations)
        early_stopping.update_loss(iter_loss)
        if early_stopping.stop_training():
            torch.save(model, r"C:\Users\denis\Documents\TTS_saved_model\net.pb")
            break

        print(iter_loss)


        # if valid_loss[-1] < min_loss:
        #     torch.save(net, os.path.join(experiments_path, "net.pb".format(n_classes)))
        #     min_loss = valid_loss[-1]
        #
        # print('Epoch %d/%d, Tr Loss: %.4f, Tr Fscore: %.4f, Val Loss: %.4f, Val Fscore: %.4f'
        #       % (epoch + 1, config.num_epochs, train_loss[-1], train_fscore[-1],
        #          valid_loss[-1], valid_fscore[-1]))
        #
        # with open(os.path.join(experiments_path, "loss_track.pkl"), 'wb') as f:
        #     pickle.dump([train_loss, train_fscore, valid_loss, valid_fscore], f)



