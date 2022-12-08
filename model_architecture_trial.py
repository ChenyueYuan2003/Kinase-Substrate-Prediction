"""
# -*- coding = utf-8 -*-
#!/usr/bin/env python
# @Project : kspred
# @File : model_architecture_trial.py
# @Author : ycy
# @Time : 2022/12/7 11:37
# @Software : PyCharm Professional
"""
import nni
import numpy as np
import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
import torchmetrics
from nni.retiarii import model_wrapper
from train import get_data
from nni.retiarii.evaluator import FunctionalEvaluator


@model_wrapper  # this decorator should be put on the out most
class ModelSpace(nn.Module):
    def __init__(self):
        super().__init__()
        feature1 = nn.ValueChoice([2048, 3072, 4096])
        feature2 = nn.ValueChoice([1024, 1536, 2048])
        feature3 = nn.ValueChoice([128, 256, 512])
        self.fc1 = nn.Linear(256, feature1)
        self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))
        self.fc2 = nn.Linear(feature1, feature2)
        self.dropout2 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))
        self.fc3 = nn.Linear(feature2, feature3)
        self.fc4 = nn.Linear(feature3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        output = F.sigmoid(x)
        return output


def train_test_split(x, y, train_size=0.8):
    dataset = torch.utils.data.TensorDataset(x, y)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(train_size * len(dataset)),
                                                                          len(dataset) - int(
                                                                              train_size * len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def train(model, train_loader, optimizer, loss_f):
    model.train()
    for batch_index, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(x_batch)
        loss = loss_f(out, y_batch)
        loss.backward()
        optimizer.step()


def test(model, test_loader, loss_f, device):
    model.eval()
    test_loss_record = []
    acc_test = torchmetrics.Accuracy().to(device)

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = model.forward(x_batch)
            test_loss = loss_f(pred, y_batch)
            test_loss_record.append(test_loss.item())
            acc_test.update(pred, y_batch.int())
    acc_val = acc_test.compute()
    acc_test.reset()
    return float(acc_val)


# print(model_space)


def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x_train, y_train = get_data(device, './data/samples/', 0)
    train_loader, test_loader = train_test_split(x_train, y_train, train_size=0.8)

    model = model_cls()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_f = torch.nn.BCELoss()

    for epoch in range(40):
        # train the model for one epoch
        train(model, train_loader, optimizer, loss_f)
        # test the model for one epoch
        accuracy = test(model, test_loader, loss_f, device)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)
    # report final test result
    nni.report_final_result(accuracy)


import nni.retiarii.strategy as strategy

search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted
model_space = ModelSpace()
evaluator = FunctionalEvaluator(evaluate_model)
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'kspred_search'
exp_config.max_trial_number = 250
exp_config.trial_concurrency = 1
exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True
exp_config.training_service.max_trial_number_per_gpu = 2
exp.run(exp_config, 8081)
