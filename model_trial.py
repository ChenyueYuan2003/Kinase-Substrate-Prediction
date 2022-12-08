"""
# -*- coding = utf-8 -*-
#!/usr/bin/env python
# @Project : kspred
# @File : model_trial.py
# @Author : ycy
# @Time : 2022/12/5 18:14
# @Software : PyCharm Professional
"""

import torchmetrics
import torch.nn as nn

import numpy as np

import torch

import nni

from train import get_data


class Net(torch.nn.Module):
    def __init__(self, activation):
        super(Net, self).__init__()
        if activation == 'relu':
            self.f = nn.Sequential(
                nn.Linear(256, 2048),
                nn.ReLU(),
                nn.Dropout(0.75),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(0.75),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 1))
        elif activation == 'tanh':
            self.f = nn.Sequential(
                nn.Linear(256, 2048),
                nn.Tanh(),
                nn.Dropout(0.75),
                nn.Linear(2048, 1024),
                nn.Tanh(),
                nn.Dropout(0.75),
                nn.Linear(1024, 128),
                nn.Tanh(),
                nn.Linear(128, 1))
        elif activation == 'sigmoid':
            self.f = nn.Sequential(
                nn.Linear(256, 2048),
                nn.Sigmoid(),
                nn.Dropout(0.75),
                nn.Linear(2048, 1024),
                nn.Sigmoid(),
                nn.Dropout(0.75),
                nn.Linear(1024, 128),
                nn.Sigmoid(),
                nn.Linear(128, 1))

    def forward(self, x):
        x = self.f(x)
        x = torch.sigmoid(x)
        return x

    def predict(self, x):
        x = self.f(x)
        x = torch.sigmoid(x)
        return x


def train_test_split(x, y, train_size=0.8):
    dataset = torch.utils.data.TensorDataset(x, y)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(train_size * len(dataset)),
                                                                          len(dataset) - int(
                                                                              train_size * len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)
    return train_loader, test_loader


def train(model, train_loader, optimizer, loss_f):
    model.train()
    for batch_index, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(x_batch)
        loss = loss_f(out, y_batch)
        loss.backward()
        optimizer.step()


def test(model, test_loader, loss_f):
    model.eval()
    test_loss_record = []
    acc_test = torchmetrics.Accuracy().to(device)

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = model.predict(x_batch)
            test_loss = loss_f(pred, y_batch)
            test_loss_record.append(test_loss.item())
            acc_test.update(pred, y_batch.int())
    acc_val = acc_test.compute()
    acc_test.reset()
    return float(acc_val), np.mean(test_loss_record)


params = {

    "batch_size": 32,
    "last_hidden_size": 128,
    "lr": 0.001,
    "activation": 'relu'

}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device : {device}")
model = Net(params['activation']).to(device)
loss_f = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

x_train, y_train = get_data(device, './data/samples/', 0)
train_loader, test_loader = train_test_split(x_train, y_train, train_size=0.8)

for epoch in range(40):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train(model, train_loader, optimizer, loss_f)
    acc_val, test_loss = test(model, test_loader, loss_f)
    nni.report_intermediate_result(acc_val)
nni.report_final_result(acc_val)

# for x_batch, y_batch in train_loader:
#     print(x_batch.shape, y_batch.shape)
#     print(x_batch[:10])
#     print(y_batch[:10])
#     break
# print('------------------------------------')
# for x, y in test_loader:
#     print(x.shape, y.shape)
#     print(x[:10])
#     print(y[:10])
#     break


#
#     # t_end = time.time()
#     # if (epoch + 1) % 1 == 0:
#     #     logger.info(
#     #         'Repeat number {}/{}, Fold number {} / {}, Epoch {} / {} '.format(repeat + 1, data_repeat,
#     #                                                                           fold + 1,
#     #                                                                           kfold.get_n_splits(),
#     #                                                                           epoch + 1, epochs))
#     #     logger.info(
#     #         f"epoch:{epoch + 1}, epoch time: {t_end - t_start:.5}, avg_train_loss: {np.mean(loss_record):.8f}, avg_test_loss: {np.mean(test_loss_record):.8f}")
#     #
#     #     logger.info(
#     #         f"aur: {val_metrics['auroc']:.5f}, pre: {val_metrics['prec']:.5f}, rec: {val_metrics['rec']:.5f},acc: {val_metrics['acc_test']:.5f}")
#     # # # 早停止
#     # early_stopping(np.mean(test_loss_record), model)
#     # # 达到早停止条件时，early_stop会被置为True
#     # if early_stopping.early_stop:
#     #     logger.info("---------Early stopping--------")
#
#     # break  # 跳出迭代，结束训练
#
#     # if (epoch + 1) % epochs == 0:
#     #     metrics['acc'].append(torch.tensor([val_metrics['acc_test']], device=device))
#     #     metrics['precision'].append(torch.tensor([val_metrics['prec']], device=device))
#     #     metrics['recall'].append(torch.tensor([val_metrics['rec']], device=device))
#     #     metrics['auroc'].append(torch.tensor([val_metrics['auroc']], device=device))
#     metric_collection.reset()
#
# # torch.save(model.state_dict(), model_path + f'model_{repeat + 1}_{fold + 1}.pth')
#
# report final result

# # logger.info('Final result is %g', val_metrics['acc_test'])
# # logger.info('Send final result done.')
