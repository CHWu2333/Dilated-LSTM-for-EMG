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


def arg_parser():
    bool_option = ['True', 'False']
    measurement_option = ['min', 'max']
    scheduler_option = ['plateau']
    optimizer_option = ['sgd', 'adam', 'adamw']

    parser = argparse.ArgumentParser(description='drnn')

    # args for modeling
    parser.add_argument('--window_size', '-w', type=int, default=600, metavar='',
                        help='sampling window size, default is 600, which is 300ms')
    parser.add_argument('--step_size', '-s', type=int, default=20, metavar='',
                        help='sampling window rolling time step, default is 20, which is 10ms')
    parser.add_argument('--batch_size', '-b', type=int, default=128, metavar='',
                        help='batch size of input signal, default is 12')
    parser.add_argument('--dilation', '-d', type=int, default=0, metavar='',
                        help='dilation rate, default is 0, which is no dilation')
    parser.add_argument('--num_layers', '-n', type=int, default=3, metavar='',
                        help='the layer number of LSTM')
    parser.add_argument('--z_score_norm', '-z', choices=bool_option, type=str, default=True, metavar='',
                        help='Whether using z-score normalization to preprocess the data')
    parser.add_argument('--learning_rate', '-l', type=float, default=0.001, metavar='',
                        help='initial learning rate')
    parser.add_argument('--hidden_size', '-H', type=int, default=128, metavar='',
                        help='number of hidden units')
    parser.add_argument('--input_size', '-I', type=int, default=12, metavar='',
                        help='number of input_dimension, fixed to 12')
    parser.add_argument('--num_kernels', '-k', type=int, default=(32, 64, 64, 128, 128, 256, 256), metavar='',
                        help='list of CNN kernels number')
    parser.add_argument('--list_dilation', '-i', type=int, default=(1, 2, 4, 8, 8, 8, 1), metavar='',
                        help='list of dilation of CNN networks')

    # args for the lr_scheduler(use plateau)
    parser.add_argument('--factor', type=float, default=0.5, metavar='', help="lr scheduler's factor")
    parser.add_argument('--patience', type=int, default=50, metavar='', help="lr scheduler's patience")
    parser.add_argument('--threshold', type=float, default=1e-2, metavar='', help="lr scheduler's threshold")
    parser.add_argument('--lr_limit', type=float, default=1e-4, metavar='', help="lr scheduler's lr limit")
    parser.add_argument('--measurement', '-m', choices=measurement_option, default='min', metavar='',
                        help="lr scheduler's measurement")
    parser.add_argument('--scheduler', '-S', choices=scheduler_option, default='plateau', metavar='',
                        help='learning rate scheduler, default is plateau')
    parser.add_argument('--optimizer', '-o', default='adamw', choices=optimizer_option, metavar='', help='optimizer')
    parser.add_argument('--weight_decay', '-W', default=1e-2, metavar='', help='regularization: L2 penalty')

    # args for training
    parser.add_argument('--progress_bar', '-p', choices=bool_option, type=str, default=False, metavar='',
                        help='show progress bar when training')
    parser.add_argument('--cuda', choices=bool_option, type=str, default=True, metavar='',
                        help='use GPU for the training')
    parser.add_argument('--test', choices=bool_option, type=str, default=False, metavar='',
                        help='take a test run, skip waiting time for training, check bugs in code')
    parser.add_argument('--plot', choices=bool_option, type=str, default=False, metavar='', help='plot training curves')
    parser.add_argument('--save', choices=bool_option, type=str, default=False, metavar='',
                        help='save the trained model')
    parser.add_argument('--training_set', '-train', type=int, default=(1, 2), metavar='',
                        help='training set, select from 1-41, default is [1, 2]')
    parser.add_argument('--testing_set', '-test', type=int, default=(3, 4), metavar='',
                        help='training set, select from 1-41, default is [3, 4], # of testing set must be over 2')
    parser.add_argument('--epoch', '-e', type=int, default=20, metavar='', help='training epochs')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.weight_decay != 'b':
        args.weight_decay = float(args.weight_decay)
    return args


def _rnn_reformat(x, input_dims, n_steps):
    """
    This function reformat input to the shape that standard RNN can take.

    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # permute batch_size and n_steps
    x_ = np.transpose(x, [1, 0, 2])
    # reshape to (n_steps*batch_size, input_dims)
    x_ = np.reshape(x_, [-1, input_dims])
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = np.split(x_, n_steps, 0)

    return x_reformat


class SingleLSTM(nn.Module):
    """
    window_size = n_steps = 600, input_size = input_dimension = 12
    """

    def __init__(self, args, input_size, hidden_size):
        super(SingleLSTM, self).__init__()
        self.window_size = args.window_size
        self.num_layers = args.num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dilation = args.dilation
        self.dilated_n_steps = self.window_size // (self.dilation + 1)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

    def forward(self, x):
        """
        x shape (batch, n_step, input_size)
        out shape (batch, n_step, input_size)
        """
        # print('input_lstm_shape:')
        # print(np.shape(x))
        # dilated_inputs = x[:, self.dilation, :]
        # for i in range(1, self.dilated_n_steps):
        #     dilated_inputs = torch.cat([dilated_inputs, x[:, (i + 1) * (self.dilation + 1) - 1, :]], dim=1)
        # x = np.array(x, dtype=np.float32)
        # if self.dilation > 0:
        #     i = range(1, self.dilated_n_steps+1)
        #     dilated_inputs = x[:, i * (self.dilation + 1) - 1, :]
        # else:
        #     dilated_inputs = x
        # dilated_inputs = torch.from_numpy(dilated_inputs)
        dilated_inputs = torch.tensor(x)
        # print('dilated input shape:')
        # print(np.shape(dilated_inputs))
        out, _ = self.lstm(dilated_inputs)
        # print('lstm self.dilated_n_steps:', self.dilated_n_steps)
        # print('output shape:')
        # print(np.shape(out))
        return out


class CnnBlock(nn.Module):
    """
    1D CNN modules for feature extraction
    Input signal is (batch_size, LSTM_Out_size, input_channel)
    """

    def __init__(self, input_size, kernel_size, num_kernels, dilation):
        super(CnnBlock, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_kernels, kernel_size=kernel_size,
                               dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(num_features=num_kernels)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        # print('input cnn shape', np.shape(x))
        x = self.conv1(x)
        x = self.batch_norm(x)
        out = self.PReLU(x)
        return out


class FullConnection(nn.Module):
    def __init__(self):
        super(FullConnection, self).__init__()
        self.Flatten = nn.Flatten(1, 2)
        self.Dense1 = nn.Linear(120832, 64)
        self.Dense2 = nn.Linear(64, 32)
        self.Dense3 = nn.Linear(32, 17)

    def forward(self, x):
        out = self.Flatten(x)
        out = self.Dense1(out)
        out = self.Dense2(out)
        out = self.Dense3(out)
        return out


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.SingleLSTM1 = SingleLSTM(self.args, input_size=12, hidden_size=128)
        # self.SingleLSTM2 = SingleLSTM(self.args, input_size=128, hidden_size=128)
        # self.SingleLSTM3 = SingleLSTM(self.args, input_size=128, hidden_size=128)
        # self.SingleLSTM4 = SingleLSTM(self.args, input_size=128, hidden_size=128)
        self.CNN_list = []
        for j in range(len(self.args.num_kernels)):
            if j == 0:
                self.Cnn = CnnBlock(input_size=128, kernel_size=5,
                                    num_kernels=self.args.num_kernels[j], dilation=self.args.list_dilation[j])
                self.CNN_list.append(self.Cnn)
            else:
                self.Cnn = CnnBlock(input_size=self.args.num_kernels[j-1], kernel_size=5,
                                    num_kernels=self.args.num_kernels[j], dilation=self.args.list_dilation[j])
                self.CNN_list.append(self.Cnn)
        self.FullConnection = FullConnection()

    def forward(self, x):

        # for i in range(self.args.num_layers):
        #     if i == self.args.num_layers-1:
        #         self.SingleLSTM = SingleLSTM(self.args, hidden_size=128)
        #         x = self.SingleLSTM(x)
        #     else:
        #         self.SingleLSTM = SingleLSTM(self.args, hidden_size=12)
        #         x = self.SingleLSTM
        x = self.SingleLSTM1(x)
        # x = self.SingleLSTM2(x)
        # x = self.SingleLSTM3(x)
        # x = self.SingleLSTM4(x)
        x = np.transpose(x, [0, 2, 1])
        # print('CNN blocks num:', len(self.CNN_list))
        for i, k in enumerate(self.CNN_list):
            # print('problem in CNN:', i)
            x = k(x)
        out = self.FullConnection(x)
        return out


def DataPreprocessing(args, data_set):
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
        self.train_data, self.train_label = DataPreprocessing(self.args, self.args.training_set)
        self.test_data, self.test_label = DataPreprocessing(self.args, self.args.testing_set)
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

        self.train_loader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=self.args.batch_size)
        self.test_loader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=self.args.batch_size)
        t2 = time.time()
        print('Data loaded! Time used:', t2 - t1)

    def create_network(self):
        """
        constructing the whole network,
        including a 3 or 4 layers of dilated LSTM, 7 modules of CNN,
         and 3 layers dense connection
         Loss
        """
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

        # # plot vertical lines for lr changing
        # vline = False
        # for i, v in enumerate(self.lr):
        #     if i >= 1 and v != self.lr[i - 1]:
        #         if not vline:
        #             plt.axvline(i + 1, linestyle='--', linewidth=1, label='lr reduced')
        #         else:
        #             plt.axvline(i + 1, linestyle='--', linewidth=1)
        #         vline = True

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
        table = arg_parser()
        Project(table).main()
    except Exception as e:
        print('--error occurs, Stopping')
        traceback.print_exc(e)
