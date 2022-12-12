import os
import time

from sklearn.model_selection import KFold
import torch.nn as nn

import logging

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
import math
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC

import train_param_parsing

# config logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
search_space = {

    "batch_size": [32, 64, 128, 256, 512],
    "lr": [0.0001, 0.001, 0.01, 0.1],
    "activation": ['relu', 'tanh', 'sigmoid']

}


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


class EarlyStopping:

    def __init__(self, mode='min', patience=7, verbose=False, delta=0, trace_func=logger.info):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = 'path'
        self.trace_func = trace_func
        self.mode = mode

        self.__value = -math.inf if mode == 'max' else math.inf

    def __call__(self, metrics, model):

        if self.best_score is None:
            self.best_score = metrics
            # self.save_checkpoint(val_loss, model)
        elif (self.mode == 'min' and metrics >= self.best_score + self.delta) or (
                self.mode == 'max' and metrics <= self.best_score + self.delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metrics
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train(model, fold_num, epochs, batch_size, optimizer, loss_f, device, data_path, data_repeat, model_path,
          optimization_num=None):
    logger.info('Training started...')
    logger.info(f"using device: {device}")
    # store the  accuracy, precision,recall for plotting
    metrics = {
        'acc': [],
        'precision': [],
        'recall': [],
        'auroc': []
    }

    for repeat in range(data_repeat):
        x_train, y_train = get_data(device, data_path, repeat)
        kfold = KFold(n_splits=fold_num, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(kfold.split(x_train, y_train)):
            # Dividing data into folds
            x_train_fold = x_train[train_index]
            x_test_fold = x_train[test_index]
            y_train_fold = y_train[train_index]
            y_test_fold = y_train[test_index]

            train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
            test = torch.utils.data.TensorDataset(x_test_fold, y_test_fold)
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

            # reinstantiate the model
            model.apply(init_weights)
            model.to(device)

            early_stopping = EarlyStopping(mode='min', patience=20)

            for epoch in range(epochs):
                t_start = time.time()

                loss_record = []
                model.train()
                for batch_index, (x_batch, y_batch) in enumerate(train_loader):
                    optimizer.zero_grad()
                    out = model(x_batch)
                    loss = loss_f(out, y_batch)
                    loss_record.append(loss.item())
                    loss.backward()
                    optimizer.step()

                model.eval()
                test_loss_record = []
                # auroc = torchmetrics.AUROC(pos_label=1).to(device)
                # precision = torchmetrics.Precision().to(device)
                # recall = torchmetrics.Recall().to(device)
                metric_collection = MetricCollection({
                    'acc_test': Accuracy().to(device),
                    'prec': Precision().to(device),
                    'rec': Recall().to(device),
                    'auroc': AUROC(pos_label=1).to(device)
                })
                with torch.no_grad():
                    for x_batch, y_batch in test_loader:
                        pred = model.predict(x_batch)
                        test_loss = loss_f(pred, y_batch)
                        test_loss_record.append(test_loss.item())
                        # acc_test.update(pred, y_batch.int())
                        # auroc.update(pred, y_batch.int())
                        # precision.update(pred, y_batch.int())
                        # recall.update(pred, y_batch.int())

                        metric_collection.forward(pred, y_batch.int())

                val_metrics = metric_collection.compute()
                metric_collection.reset()

                t_end = time.time()
                if (epoch + 1) % 1 == 0:
                    logger.info(
                        'Optimization number {}, Repeat number {} / {}, Fold number {} / {}, Epoch {} / {} '.format(
                            optimization_num + 1, repeat + 1, data_repeat,
                            fold + 1,
                            kfold.get_n_splits(),
                            epoch + 1, epochs))
                    logger.info(
                        f"epoch:{epoch + 1}, epoch time: {t_end - t_start:.5}, avg_train_loss: {np.mean(loss_record):.8f}, avg_test_loss: {np.mean(test_loss_record):.8f}")

                    logger.info(
                        f"aur: {val_metrics['auroc']:.5f}, pre: {val_metrics['prec']:.5f}, rec: {val_metrics['rec']:.5f},acc: {val_metrics['acc_test']:.5f}")
                # # 早停止
                early_stopping(np.mean(test_loss_record), model)
                # # 达到早停止条件时，early_stop会被置为True
                if early_stopping.early_stop:
                    logger.info("---------Early stopping--------")
                    metrics['acc'].append(torch.tensor([val_metrics['acc_test']], device=device))
                    metrics['precision'].append(torch.tensor([val_metrics['prec']], device=device))
                    metrics['recall'].append(torch.tensor([val_metrics['rec']], device=device))
                    metrics['auroc'].append(torch.tensor([val_metrics['auroc']], device=device))
                    # torch.save(model.state_dict(), model_path + f'model_{repeat + 1}_{fold + 1}.pth')
                    break

                # if (epoch + 1) % epochs == 0:
                #     metrics['acc'].append(torch.tensor([val_metrics['acc_test']], device=device))
                #     metrics['precision'].append(torch.tensor([val_metrics['prec']], device=device))
                #     metrics['recall'].append(torch.tensor([val_metrics['rec']], device=device))
                #     metrics['auroc'].append(torch.tensor([val_metrics['auroc']], device=device))
                # metric_collection.reset()

            # torch.save(model.state_dict(), model_path + f'model_{repeat + 1}_{fold + 1}.pth')

    metrics['acc'] = torch.cat(metrics['acc']).to('cpu').numpy()
    metrics['precision'] = torch.cat(metrics['precision']).to('cpu').numpy()
    metrics['recall'] = torch.cat(metrics['recall']).to('cpu').numpy()
    metrics['auroc'] = torch.cat(metrics['auroc']).to('cpu').numpy()

    repeat = []
    for i in range(50):
        repeat.append(math.ceil((i + 1) / 5))
    metrics['repeat'] = repeat
    # metrics = pd.DataFrame.from_dict(metrics, orient='index')
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(f'./metrics_2/metrics_{optimization_num}.csv', index=False)

    # return metrics


def get_data(device, data_path, repeat):
    positive_exm = np.loadtxt(data_path + 'pos_emb.csv', delimiter=',', unpack=True).reshape(-1, 256)
    negative_exm = np.loadtxt(data_path + f'ran_neg_emb_{repeat}.csv', delimiter=',', unpack=True).reshape(-1, 256)
    positive_exm = torch.from_numpy(positive_exm).float().to(device)
    negative_exm = torch.from_numpy(negative_exm).float().to(device)

    positive_label = torch.ones(positive_exm.shape[0], device=device)
    negative_label = torch.zeros(negative_exm.shape[0], device=device)

    x_train = torch.cat((positive_exm, negative_exm), 0).reshape(-1, 256)
    y_train = torch.cat((positive_label, negative_label), 0).reshape(-1, 1)

    return x_train, y_train


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def plot(metrics):
    metrics['acc'] = torch.cat(metrics['acc']).to('cpu')
    metrics['precision'] = torch.cat(metrics['precision']).to('cpu')
    metrics['recall'] = torch.cat(metrics['recall']).to('cpu')
    metrics['auroc'] = torch.cat(metrics['auroc']).to('cpu')

    repeat = []
    for i in range(50):
        repeat.append(math.ceil((i + 1) / 5))
    metrics['repeat'] = repeat
    metrics = pd.DataFrame(metrics)
    metrics.to_csv('./result/metrics.csv', index=False)

    sns.set_theme(style="whitegrid")
    plt.subplot(2, 2, 1)
    sns.boxplot(y='auroc', x='repeat', data=metrics, palette="Set3")
    plt.xlabel('')

    # --------------
    plt.subplot(2, 2, 2)
    sns.boxplot(y='acc', x='repeat', data=metrics, palette="Set3")
    plt.xlabel('')

    # --------------
    plt.subplot(2, 2, 3)
    sns.boxplot(y='precision', x='repeat', data=metrics, palette="Set3")
    plt.xlabel('')

    # --------------
    plt.subplot(2, 2, 4)
    sns.boxplot(y='recall', x='repeat', data=metrics, palette="Set3")
    plt.xlabel('')

    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    # savefig
    plt.savefig('./result/boxplot.png', dpi=300)
    plt.show()


def get_search_space_size(search_space):
    search_space_size = 1
    for key in search_space.keys():
        search_space_size *= len(search_space[key])
    return search_space_size


def get_next_parameters(search_space, index):
    search_space_size = get_search_space_size(search_space)
    if index >= search_space_size:
        return None
    else:
        search_space_index = index
        search_space_next = {}
        for key in search_space.keys():
            search_space_next[key] = search_space[key][search_space_index % len(search_space[key])]
            search_space_index = search_space_index // len(search_space[key])
        return search_space_next


def main():
    # get arguments
    args = train_param_parsing.get_args()
    logger.info(args)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    search_space_size = get_search_space_size(search_space)
    for i in range(search_space_size)[23:]:
        params = get_next_parameters(search_space, i)
        if params is None:
            break
        else:
            logger.info(params)

            model = Net(params['activation'])
            # model = Net(last_hidden_size=128)
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_f = torch.nn.BCELoss()

            train(model, fold_num=args.fold_num, epochs=args.epochs, batch_size=params['batch_size'],
                  data_repeat=args.data_repeat,
                  data_path=args.data_path, model_path=args.model_path,
                  optimizer=optimizer, loss_f=loss_f,
                  device=device, optimization_num=i)

    # if not os.path.exists('./result'):
    #     os.makedirs('./result')
    # plot(metrics)


if __name__ == '__main__':
    main()
