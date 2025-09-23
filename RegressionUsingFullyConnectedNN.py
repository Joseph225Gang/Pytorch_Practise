import torch
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn

data = pd.read_csv('bike_sharing.csv', index_col=0)

print(data.head())
print(data.shape)

data = pd.get_dummies(data, columns= ['season'])
print(data.sample(5))

columns = ['registered','holiday','workingday','weather','temp','atemp','season_1','season_2','season_3','season_4']

features = data[columns]

print(features.head())

target = data[['count']]

from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(features, target, test_size=0.2)

X_train = pd.get_dummies(X_train)
x_test = pd.get_dummies(x_test)
Y_train = pd.get_dummies(Y_train)
y_test = pd.get_dummies(y_test)

X_train = X_train.astype(float)
x_test = x_test.astype(float)
Y_train = Y_train.astype(float)
y_test = y_test.astype(float)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)

import torch.utils.data as data_utils
train_data = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = data_utils.DataLoader(train_data, batch_size=100, shuffle=True)

print(len(train_loader))

features_batch, target_batch = iter(train_loader).__next__()

inp = X_train_tensor.shape[1]
out = 1

hid = 10

loss_fn = torch.nn.MSELoss()

model = torch.nn.Sequential(torch.nn.Linear(inp, hid), torch.nn.Linear(hid,out))

x = torch.zeros(10, inp)
y = model(x)
writer = SummaryWriter()
dummy_input = torch.zeros(10, inp)
writer.add_graph(model, dummy_input)
writer.close()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

total_step = len(train_loader)

num_epochs = 2000

for epoch in range(num_epochs + 1) :
    for i, (features, target) in enumerate(train_loader):

        output = model(features)
        loss = loss_fn(output, target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


model.eval()

with torch.no_grad():
    y_pred = model(x_test_tensor)

print(sklearn.metrics.r2_score(y_test, y_pred))

