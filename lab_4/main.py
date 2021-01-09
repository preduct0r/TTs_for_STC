import pandas as pd
import torch
from torch import nn, optim
import torch.cuda as cuda

from sklearn.model_selection import cross_val_score, train_test_split, KFold

from lab_3.model import *
from lab_4.model import *

config = Config(lr=2e-5, batch_size=100, num_epochs=100)

df = pd.read_csv(r'C:\Users\denis\PycharmProjects\TTs_for_STC\data\features_lab4.csv')

X = pd.get_dummies(df.drop(columns=['word', 'phoneme', 'allophone',
                        'c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12'])).values.astype(np.float32)
y = df.loc[:,['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12']].values.astype(np.float32)


x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

batcher_train = DataLoader(My_Dataset2(x_train, y_train), batch_size=config.batch_size)
batcher_val = DataLoader(My_Dataset2(x_val, y_val), batch_size=config.batch_size, shuffle=False)

model = LinearNet()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.001)
early_stopping = EarlyStopping()

if cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
model.to(device)
min_loss = 1000

#собственно тренируем модель
for epoch in range(config.num_epochs):
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

        outputs= model(items)

        loss = criterion(outputs, classes)
        iter_loss += loss.item()
        loss.backward()
        optimizer.step()

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

    for i, (items, classes) in enumerate(batcher_val):
        if classes.shape[0] != config.batch_size:
            break

        items = items.to(device)
        classes = classes.to(device)

        outputs = model(items)
        loss = criterion(outputs, classes)
        iter_loss += loss.item()

        total += 1

    print('          Val loss: {}, accuracy: {}\n'.format(np.round(iter_loss/total, 4), np.round(correct/total/config.batch_size, 2)))

    early_stopping.update_loss(iter_loss)
    if early_stopping.stop_training():
        torch.save(model, r"C:\Users\denis\Documents\TTS_saved_model\lab_4_net.pb")
        break
