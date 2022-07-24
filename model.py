import pandas as pd
from pymatreader import read_mat
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from sklearn import preprocessing
import argparse
import time
import matplotlib.pyplot as plt
import traceback


class Args:
    window_size = 600
    step_size = 20
    batch_size = 128
    dilation = 0
    num_layers = 3
    z_score_norm = True
    hidden_size = 128
    input_size = 12
    num_kernels = (32, 64, 64, 128, 128, 256, 256)
    list_dilation = (1, 2, 4, 8, 8, 8, 1)

    learning_rate = 0.001
    factor = 0.5
    patience = 50
    threshold = 1e-2
    lr_limit = 1e-4
    measurement = 'min'
    optimizer = 'adam'
    weight_decay = 1e-2
    progress_bar = True
    cuda = True
    test = False
    plot = False
    save = False
    scheduler = 'plateau'
    weight_decay = 1e-2

    training_set = (1, 2)
    testing_set = (3, 4)
    epoch = 20
    cuda = cuda and torch.cuda.is_available()


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # single LSTM
        self.window_size = args.window_size
        self.num_layers = args.num_layers
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.dilation = args.dilation
        self.dilated_n_steps = self.window_size // (self.dilation + 1)
        self.lstm1 = nn.LSTM(input_size=12, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        # CNN Blocks
        # num_kernels = (32, 64, 64, 128, 128, 256, 256)
        # list_dilation = (1, 2, 4, 8, 8, 8, 1)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=5, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, dilation=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, dilation=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, dilation=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=1)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, dilation=1)
        self.conv7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, dilation=1)
        self.batch_norm1 = nn.BatchNorm1d(num_features=32)
        self.batch_norm2 = nn.BatchNorm1d(num_features=64)
        self.batch_norm3 = nn.BatchNorm1d(num_features=64)
        self.batch_norm4 = nn.BatchNorm1d(num_features=128)
        self.batch_norm5 = nn.BatchNorm1d(num_features=128)
        self.batch_norm6 = nn.BatchNorm1d(num_features=256)
        self.batch_norm7 = nn.BatchNorm1d(num_features=256)
        self.PReLU1 = nn.PReLU()
        self.PReLU2 = nn.PReLU()
        self.PReLU3 = nn.PReLU()
        self.PReLU4 = nn.PReLU()
        self.PReLU5 = nn.PReLU()
        self.PReLU6 = nn.PReLU()
        self.PReLU7 = nn.PReLU()

        # Full connection blocks
        self.Flatten = nn.Flatten(1, 2)
        self.Dense1 = nn.Linear(120832, 64)
        self.Dense2 = nn.Linear(64, 32)
        self.Dense3 = nn.Linear(32, 17)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = np.transpose(x, [0, 2, 1])

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.PReLU1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.PReLU2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.PReLU3(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.PReLU4(x)
        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.PReLU5(x)
        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = self.PReLU6(x)
        x = self.conv7(x)
        x = self.batch_norm7(x)
        x = self.PReLU7(x)

        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.Dense2(x)
        out = self.Dense3(x)

        return out


def data_preprocessing(args, data_set):
    """
    This function is called when loading the data, it takes a lot of time.
    The function is will turn all E1_A1 data to a (x, window_size, 2 + channel_num)
    In which, x is the number of all pieces of windowing,
    window size in paper is set to be 600, which is 300ms
    step_size in paper is set to be 20 which is 10 ms
    channel_num in dataset is 12, fixed.
    Outputs:
        Input: EMG data with 12 channels, shape:(x, window_size, channel_num)
        label: EMG corresponding restimulus digit, shape:(x,)
    """
    data_dir = '/scratch/cw3755/my_project/data_E1/'
    inputs = []
    labels = []
    for i in data_set:
        f = data_dir + '/S' + str(i) + '_E1_A1.mat'
        data_raw = read_mat(f)
        emg = preprocessing.scale(data_raw['emg'])
        df1 = pd.DataFrame(emg)
        df2 = pd.DataFrame(data_raw['restimulus'])
        df3 = pd.DataFrame(data_raw['repetition'])
        df = pd.concat([df3, df2, df1], axis=1)
        for repetition in range(1, 7):
            for restimulus in range(1, 18):
                df4 = df.loc[df.iloc[:, 0] == repetition, :]
                df5 = df4.loc[df.iloc[:, 0] == restimulus, :]
                for step in range((df5.shape[0] - args.window_size) // args.step_size):
                    inputs.append(
                        df5.iloc[args.step_size * step:args.step_size * step + args.window_size, 2:].values)
                    labels.append(restimulus)
    inputs = np.array(inputs)
    labels = np.array(labels)
    return inputs, labels


class Project:
    def __init__(self, args):
        self.args = args
        self.train_loader = self.test_loader = None
        self.net = self.loss = self.scheduler = None
        self.optimizer = args.optimizer
        self.train_loss = []
        self.test_loss = []
        self.test_acc = []
        self.lr = []
        self.trainable_parameters = 0

    def process_data(self):
        """
        preprocess and load data
        """
        print('loading data...')
        t1 = time.time()
        self.train_data, self.train_label = data_preprocessing(self.args, self.args.training_set)
        self.test_data, self.test_label = data_preprocessing(self.args, self.args.testing_set)
        # print(np.shape(self.train_data))
        # print(np.shape(self.train_label))
        train = [[self.train_data[i, :, :], self.train_label[i]] for i in range(len(self.train_label))]
        test = [[self.test_data[i, :, :], self.test_label[i]] for i in range(len(self.test_label))]
        # print(np.shape(train))
        # print(np.shape(test))

        self.train_data = torch.tensor(self.train_data)
        self.test_data = torch.tensor(self.test_data)
        self.train_label = torch.tensor(self.train_label)
        self.test_label = torch.tensor(self.test_label)

        self.train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=self.args.batch_size)
        self.test_loader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=self.args.batch_size)
        t2 = time.time()
        print('Data loaded! Time used:', t2 - t1)

    def create_network(self):
        self.net = Model(self.args)
        if self.args.cuda:
            self.net = self.net.cuda()
        self.loss = nn.CrossEntropyLoss()
        if self.args.weight_decay == 'b':
            optimizer_table = {
                'sgd': optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=0.9, nesterov=True,
                                 weight_decay=1e-5 * self.args.batch_size),
                'adam': optim.Adam(self.net.parameters(), lr=self.args.learning_rate, amsgrad=True,
                                   weight_decay=1e-5 * self.args.batch_size),
                'adamw': optim.AdamW(self.net.parameters(), lr=self.args.learning_rate, amsgrad=True,
                                     weight_decay=1e-5 * self.args.batch_size)
            }
        else:
            optimizer_table = {
                'sgd': optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=0.9, nesterov=True,
                                 weight_decay=self.args.weight_decay),
                'adam': optim.Adam(self.net.parameters(), lr=self.args.learning_rate, amsgrad=True,
                                   weight_decay=self.args.weight_decay),
                'adamw': optim.AdamW(self.net.parameters(), lr=self.args.learning_rate, amsgrad=True,
                                     weight_decay=self.args.weight_decay)
            }

        self.optimizer = optimizer_table[self.args.optimizer.lower()]
        if self.args.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                self.args.measurement,
                threshold=self.args.threshold,
                factor=self.args.factor,
                patience=self.args.patience,
                min_lr=self.args.lr_limit
            )

    def train_model(self):
        """
        stage for training model and print performance every epoch
        save the trained model at the end
        """
        for epoch in range(1, self.args.epoch + 1):
            train_loss = 0
            test_loss = 0
            test_acc = 0
            if self.args.progress_bar:
                self.train_loader = tqdm.tqdm(self.train_loader)
                self.test_loader = tqdm.tqdm(self.test_loader)

                inputs = self.train_data
                labels = self.train_label
                if self.args.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                self.optimizer.zero_grad()
                # print('inputs shape = ', np.shape(inputs))
                predicted_output = self.net(inputs)
                fit = self.loss(predicted_output, labels)
                fit.backward()
                self.optimizer.step()
                train_loss += fit.item()
                if self.args.test:
                    break
            for i, data in enumerate(self.test_loader):
                with torch.no_grad():
                    inputs, labels = data
                    if self.args.cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    predicted_output = self.net(inputs)
                    test_acc += sum(i == j for i, j in zip(torch.argmax(predicted_output, dim=1), labels))
                    fit = self.loss(predicted_output, labels)
                    test_loss += fit.item()
                if self.args.test:
                    break
            train_loss = train_loss / len(self.train_loader)
            test_loss = test_loss / len(self.test_loader)
            test_acc = test_acc / 10000
            test_acc = test_acc.tolist()
            lr = self.optimizer.param_groups[0]['lr']

            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)
            self.test_acc.append(test_acc)
            self.lr.append(lr)
            if self.args.scheduler == 'plateau':
                if self.args.measurement == 'min':
                    self.scheduler.step(test_loss)
                else:
                    self.scheduler.step(test_acc)

            print(
                f'epoch {epoch}, train loss {train_loss:.4}, test loss {test_loss:.4}, test acc {test_acc:.4}, lr {lr:.4}')

        if self.args.save:
            torch.save(self.net.state_dict(), f"{int(time.time())}.pt")

    def plot_result(self):
        """
        stage for ploting curves
        """
        line_w = 1
        dot_w = 4
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # ax1 for train loss and test loss(left y-axis)
        # ax2 for test acc(right y-axis)
        ax1.plot(range(1, self.args.epoch + 1), self.train_loss, 'b--', linewidth=line_w, label='train error')
        ax1.plot(range(1, self.args.epoch + 1), self.test_loss, 'r--', linewidth=line_w, label='test error')
        ax2.plot(range(1, self.args.epoch + 1), self.test_acc, 'g--', linewidth=line_w, label='test accuracy')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        ax1.scatter(range(1, self.args.epoch + 1), self.train_loss, color='b', s=dot_w)
        ax1.scatter(range(1, self.args.epoch + 1), self.test_loss, color='r', s=dot_w)
        ax2.scatter(range(1, self.args.epoch + 1), self.test_acc, color='g', s=dot_w)

        # note the highest test acc
        best_acc = max(self.test_acc)
        best_acc_epoch = self.test_acc.index(max(self.test_acc))
        ax2.scatter(best_acc_epoch + 1, best_acc, color='#7D3C98', marker='x', s=10, linewidths=3)
        ax2.annotate(
            f"epoch = {best_acc_epoch}, max acc = {best_acc}",
            (best_acc_epoch + 1, best_acc + 0.003),
            ha='center',
            color='#7D3C98'
        )

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax2.set_ylabel('accuracy')

        # set some value to make the figure looks better if epoch is too large
        ax1.set_ylim([0, max(max(self.train_loss), max(self.test_loss)) + 1])
        ax2.set_ylim([min(self.test_acc) - 0.001, 0.95])
        plt.grid(True)
        plt.show()

    def main(self):
        print("-----------START----------")
        print(self.args)
        self.create_network()
        self.process_data()
        self.train_model()
        if self.args.plot:
            self.plot_result()


if __name__ == "__main__":
    try:
        print(torch.cuda.is_available())
        table = Args
        Project(table).main()
    except Exception as e:
        print('--error occurs, Stopping')
        traceback.print_exc(e)
