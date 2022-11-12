##################################################
# Author: {Xuanyu ZHENG}
# Copyright: Copyright {2021}, {online federated J-OMP-DL-based channel estimator}
# Credits: [{Xuanyu ZHENG, Vincent LAU}]
# Version: {1}.{1}.{0}
##################################################
import time
from MuMimoUtils import *
from torch.utils.data import Dataset, DataLoader
import torch
from matplotlib import pyplot as plt
import torch.nn as nn


# we  use multi-path channels to generate real-time channel data in this code
class MUMIMOData(Dataset):
    # Constructor
    def __init__(self, sys, S, F, num):
        """
         Task 1: Generate channel data for each MU. We can assume the change of support
               position (i.e., the scatterer) is much slower than fading.
         Task 1: Generate data of pilot feedbacks from each user.
               construct y = S * h + n using numpy, then convert to real tensor
         :param sys: a dictionary containing system parameters
         :param S: pilot matrix, fixed for training & validation & testing set, complex M x N
         :param F: dft matrix, fixed for training & validation & testing set, complex N x N
         :param num: total number of channels from all users,
              you can regard it as num = K * (ch uses) = batch_size * (ch_uses) (under single epoch case)
         """

        N = sys['N']        # BS tx antenna number
        M = sys['M']        # pilot length
        K = sys['K']        # number of MUs
        rho = sys['rho']    # tx power
        SNR = sys['SNR']    # per channel SNR
        lamda = sys['lamda']  # carrier wavelength
        D = sys['D']        # antenna spacing
        Pc = sys['Pc']      # common support number
        Pd = sys['Pd']      # individual support number

        P = Pc + Pd         # total number of supports
        noise_var = rho / (10.0 ** (SNR / 10.0))  # compute noise variance
        envir_num = int(K * kappa)       # number of environments experienced by the system

        # get the fading channels from all the users
        theta_grids = np.arcsin(lamda / D * (-1 / 2 + np.arange(N) / N)) + 0       # the AoD on fixed supports, N x 1 array
        # common support index
        ind_c = np.array([23, 46])
        # ind_c = ind_c[0:Pc]
        ind_remain = np.array([supp for supp in range(N) if supp not in ind_c])   # support index other than ind_c
        # individual support set matrix: envir_num x Pd, the individual support will only come from envir_num supports
        ind_d_set = np.zeros((envir_num, Pd))
        # individual support & common support index set matrix
        ind_set = np.zeros((envir_num, P))
        for envir in range(envir_num):
            ind_d_set[envir] = np.random.choice(ind_remain, size=Pd, replace=False)
            ind_set[envir] = np.sort(np.concatenate((ind_c, ind_d_set[envir])))
        ind_set = ind_set.astype(int)

        # generate channel
        h = generate_sparse_ch(sys, theta_grids, ind_set, lamda, D, num)  # shape = N x num
        h_tilde = complex_decompose(h)  # shape = 2N x num

        # compute the basis
        A = S @ F
        # get the real counterpart of measurement matrix
        A_real = np.real(A)
        A_imag = np.imag(A)
        A_row1 = np.concatenate((A_real, -A_imag), axis=1)
        A_row2 = np.concatenate((A_imag, A_real), axis=1)
        A_tilde = np.concatenate((A_row1, A_row2), axis=0)

        # get the angular sparse channel
        x = F.conj().T @ h  # shape = N x num
        x_tilde = complex_decompose(x)  # shape = 2N x num

        # get the feedback pilots
        Noise = (np.random.randn(M, num) + 1j*np.random.randn(M, num)) \
            * np.sqrt(noise_var/2)  # shape = num x Nr x Ls
        r = S @ h + Noise  # shape = M x num
        r_tilde = complex_decompose(r)

        # get the data for common support identification
        self.A = A      # the complex measurement matrix
        self.rT = r.T    # the complex pilot feedbacks

        # transform to pytorch and swap dimension (because in pytorch the first dim is the sample number)
        self.r_tilde = torch.from_numpy(r_tilde.T).float()    # real-vectorized nonlinear signal num x 2M
        self.h_tilde = torch.from_numpy(h_tilde.T).float()    # real-vectorized spatial channel
        self.x_tilde = torch.from_numpy(x_tilde.T).float()    # real-vectorized angular sparse channel
        self.A_tilde = torch.from_numpy(A_tilde.T).float()    # real-counterpart of RIP matrix
        self.len = num
        self.ind_set = ind_set
        self.ind_c = ind_c

    def __getitem__(self, index):
        return self.r_tilde[index], self.h_tilde[index], self.x_tilde[index], self.rT[index]

    # Get Length
    def __len__(self):
        return self.len

    # get the real counterpart of sparsifying matrix
    def get_A_tilde(self):
        return self.A_tilde

    # get the common & individual support set
    def get_ind_set(self):
        return self.ind_set

    # get the common support set
    def get_ind_c(self):
        return self.ind_c


# validation and training
class MUMIMODataVT(Dataset):
    # Constructor
    def __init__(self, sys, S, F, ind_set, num):
        """
         Task 1: Generate channel data for each MU， We can assume the change of support
               position (i.e., the scatterer) is much slower than fading.
         Task 1: Generate data of pilot feedbacks from each user.
               construct y = S * h + n using numpy, then convert to real tensor
         :param sys: a dictionary containing system parameters
         :param S: pilot matrix, fixed for training & validation & testing set, complex M x N
         :param F: dft matrix, fixed for training & validation & testing set, complex N x N
         :param ind_set: ind_set is the same as training set
         :param num: total number of channels from all users,
              you can regard it as num = K * (ch uses) = batch_size * (ch_uses) (under single epoch case)
         """

        N = sys['N']        # BS tx antenna number
        M = sys['M']        # pilot length
        rho = sys['rho']    # tx power
        SNR = sys['SNR']    # per channel SNR
        lamda = sys['lamda']  # carrier wavelength
        D = sys['D']        # antenna spacing

        noise_var = rho / (10.0 ** (SNR / 10.0))  # compute noise variance

        # get the fading channels from all the users
        theta_grids = np.arcsin(lamda / D * (-1 / 2 + np.arange(N) / N)) + 0

        # generate channel
        h = generate_sparse_ch(sys, theta_grids, ind_set, lamda, D, num)  # shape = N x num
        h_tilde = complex_decompose(h)  # shape = 2N x num

        # compute the basis
        A = S @ F
        # get the real counterpart of measurement matrix
        A_real = np.real(A)
        A_imag = np.imag(A)
        A_row1 = np.concatenate((A_real, -A_imag), axis=1)
        A_row2 = np.concatenate((A_imag, A_real), axis=1)
        A_tilde = np.concatenate((A_row1, A_row2), axis=0)

        # get the angular sparse channel
        x = F.conj().T @ h  # shape = N x num
        x_tilde = complex_decompose(x)  # shape = 2N x num

        # get the feedback pilots
        Noise = (np.random.randn(M, num) + 1j*np.random.randn(M, num)) \
            * np.sqrt(noise_var/2)  # shape = num x Nr x Ls
        r = S @ h + Noise  # shape = M x num
        r_tilde = complex_decompose(r)

        # transform to pytorch and swap dimension (because in pytorch the first dim is the sample number)
        self.r_tilde = torch.from_numpy(r_tilde.T).float()    # real-vectorized nonlinear signal num x 2M
        self.h_tilde = torch.from_numpy(h_tilde.T).float()    # real-vectorized spatial channel
        self.x_tilde = torch.from_numpy(x_tilde.T).float()    # real-vectorized angular sparse channel
        self.A_tilde = torch.from_numpy(A_tilde.T).float()    # real-counterpart of RIP matrix
        self.len = num
        self.ind_set = ind_set

    def __getitem__(self, index):
        return self.r_tilde[index], self.h_tilde[index], self.x_tilde[index]

    # Get Length
    def __len__(self):
        return self.len

    # get the real counterpart of sparsifying matrix
    def get_A_tilde(self):
        return self.A_tilde


# A clever way of building DNN1, where you do not need to add hidden layers manually
class NetS(nn.Module):
    def __init__(self, widths, p=0.5):
        """
        :param widths: a list contains the widths of each layer
        :param p: dropout parameter, p=0 means no dropout, default is p=0
        """
        super(NetS, self).__init__()
        self.hidden = nn.ModuleList()
        self.drop = nn.Dropout(p=p)

        for input_size, output_size in zip(widths, widths[1:]):
            linear = nn.Linear(input_size, output_size)

            self.hidden.append(linear)

        # By default, the weights are initialized by uniform distribution in [-1/sqrt(in_size), +1/sqrt(out_size)]
        # We can also use Xavier method for uniform distribution in [+-sqrt(6/(in_size+out_size))]
        # Explicitly call torch.nn.init.xavier_uniform_(linear1.weight)
        # for relu, use He initialization, calling torch.nn.init.kaiming_uniform_(linear1.weight, nonlinearity='relu')

    def forward(self, x):
        """
        :param x:   x = r_tilde: input of dimension num x 2M
        :return:
        """
        L = len(self.hidden)  # number of (hidden + output) layers
        for (l, linear) in zip(range(L), self.hidden):
            if l < L - 1:  # for hidden layers
                x = torch.relu(linear(x))
                x = self.drop(x)
            else:  # for output layer
                x = torch.sigmoid(linear(x))

        return x


# A clever way of building DNN2, where you do not need to add hidden layers manually
class Net(nn.Module):
    def __init__(self, widths, p=0.5):
        """
        :param widths: a list contains the widths of each layer
        :param p: dropout parameter, p=0 means no dropout, default is p=0
        """
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        self.drop = nn.Dropout(p=p)

        for input_size, output_size in zip(widths, widths[1:]):
            linear = nn.Linear(input_size, output_size)
            # torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden.append(linear)

    def forward(self, x):
        """
        :param x:   x = r_tilde: input of dimension num x 2M
        :return:
        """
        L = len(self.hidden)  # number of (hidden + output) layers
        for (l, linear) in zip(range(L), self.hidden):
            if l < L - 1:  # for hidden layers
                x = torch.relu(linear(x))
                x = self.drop(x)
            else:  # for output layer
                x_tilde_hat = linear(x)  # output channel, dim(x_tilde_hat) = num x 2N
                r_tilde_hat = torch.matmul(x_tilde_hat, A_tilde)  # linear output y, dim(y) = num x 2M

        return r_tilde_hat, x_tilde_hat


def train(model1, model2, criterion1, criterion2, train_loader, val_dataset,
          optimizer1, optimizer2, ind_c, epochs=2000):
    LOSS = []       # store the loss in training
    LOSS_val = []   # store the loss in validation
    ERR = []        # store the err of real-sparse vector in training
    ERR_val = []    # store the err of real-sparse vector in

    K = train_loader.batch_size
    noise_var = rho / (10.0 ** (SNR / 10.0))  # compute noise variance
    gamma = np.sqrt(noise_var / rho * M * np.log(N))   # compute regularizer constant gamma
    # gamma = np.sqrt(noise_var / rho)   # compute regularizer constant gamma
    print('gamma = ', gamma)
    itera = 0   # total number of times slots

    DNN1_step = 0       # number of weight update for DNN1
    batch_size_1 = 5   # number of time slots for DNNs to perform weight update
    data_batch1 = torch.zeros(batch_size_1, 2 * K * M)  # data for DNN1 is (batch_size_1, 2KM)
    data_batch2 = torch.zeros(batch_size_1 * K, 2 * M)  # data for DNN2 is (batch_size_1*K, 2M)
    x_tildes = torch.zeros(batch_size_1 * K, 2 * N)  # true channel for DNN2 is (batch_size_1*K, 2M)
    pretrain = 1

    for epoch in range(epochs):
        for r_tilde, h_tilde, x_tilde, rT in train_loader:
            # r_tilde.shape = K x 2M

            # collect input data for DNN1 and DNN2
            if itera % batch_size_1 != (batch_size_1 - 1):     # collect data only for DNN1 and DNN2 0~(batch_size_1-2)
                if itera % batch_size_1 == 0:  # initialize to 0 every batch_size_1 time slots
                    data_batch1 = torch.zeros(batch_size_1, 2 * K * M)
                    data_batch2 = torch.zeros(batch_size_1 * K, 2 * M)
                    x_tildes = torch.zeros(batch_size_1 * K, 2 * N)  # true channel for DNN2 is (batch_size_1*K, 2M)
                    # target1 = torch.zeros((batch_size_1, N))
                b_indx = itera % batch_size_1   # mini-batch index
                data_batch1[b_indx] = torch.reshape(r_tilde, (2 * K * M,))
                data_batch2[b_indx*K:(b_indx+1)*K, :] = r_tilde
                x_tildes[b_indx*K:(b_indx+1)*K, :] = x_tilde
                # target1[itera % batch_size_1, ind_c] = 1
            else:       # at (batch_size_1 - 1) collect the last samples and train
                data_batch1[batch_size_1 - 1] = torch.reshape(r_tilde, (2 * K * M,))
                data_batch2[(batch_size_1 - 1) * K:batch_size_1 * K, :] = r_tilde
                x_tildes[(batch_size_1 - 1) * K:batch_size_1 * K, :] = x_tilde
                # target1[itera % batch_size_1, ind_c] = 1

                # start to train the two-stage DNN
                # 1. forward pass for DNN 1
                model1.train()
                optimizer1.zero_grad()
                outputs1 = model1(data_batch1)  # common support probability of DNN1, dim = (batch_size_1, N)

                # 2. get input for DNN2 combining data_batch2 and outputs1
                outputs1_detach = outputs1.detach()     # dim = (batch_size_1, N)
                outputs1_detach_d = torch.repeat_interleave(outputs1_detach, K, dim=0)  # dim=(batch_size_1*K, N) check
                inputs2 = torch.cat((data_batch2, outputs1_detach_d), dim=1) # dim=(batch_size_1*K, N + 2M)

                # 3. forward pass for DNN 2
                model2.train()
                optimizer2.zero_grad()
                r_tildes_hat, x_tildes_hat = model2(inputs2)

                # 4. commpute the weight w for LASSO
                if pretrain == 1:
                    w = torch.ones(1, N, requires_grad=False)   # this is for pretraining
                else:
                    w = (1 - outputs1_detach.mean(dim=0))

                # 5. backward pass for DNN 2
                x_real = x_tildes_hat[:, 0:N]
                x_imag = x_tildes_hat[:, N:]
                # complex lasso loss, normalized by signal size 2 * M and sample number
                loss2 = 2 * M * criterion2(r_tildes_hat, data_batch2) + \
                       gamma / (batch_size_1 * K) * torch.sum(w * torch.sqrt(x_real ** 2 + x_imag ** 2))
                loss2.backward()
                optimizer2.step()

                # 6. find the common support and generate labels for DNN 1
                b = 0   # the first batch
                ind_c_est_b, _ = find_common_support(x_tildes_hat[b*K:(b+2)*K], thres=0.7)  # 4. from batch 0-1
                target1 = torch.zeros((batch_size_1, N))
                for b in range(batch_size_1):

                    # # for each data batch
                    x_tildes_hat_b = x_tildes_hat[b*K:(b+1)*K]  # x_tilde_hat.shape = (batch_size_1*K, 2*N)
                    # ind_c_est_b, _ = find_common_support(x_tildes_hat_b, thres=0.8)  # dim = C x 1
                    target1[b, ind_c_est_b] = 1

                # backward pass for DNN 1
                loss1 = criterion1(outputs1, target1)
                loss1.backward()
                optimizer1.step()

                DNN1_step += 1
                if DNN1_step > 300:    # start to train on weighted LASSO
                    pretrain = 0

                # track the performance
                if DNN1_step % 10 == 0:
                    # loss & err on training data for DNN2
                    model2.eval()  # this is to turn to evaluatio mode (turn off drop out)
                    r_tildes_hat, x_tildes_hat = model2(inputs2)
                    x_tildes_hat_real = x_tildes_hat[:, 0:N]
                    x_tildes_hat_imag = x_tildes_hat[:, N:]
                    loss2 = 2*M*criterion2(r_tildes_hat, data_batch2) + gamma / (batch_size_1 * K) * torch.sum(torch.sqrt(x_tildes_hat_real ** 2 + x_tildes_hat_imag ** 2))
                    LOSS.append(loss2.data.item())
                    err = CE_err(x_tildes, x_tildes_hat)
                    ERR.append(err.data.item())

                    # common support est accuracy on training set for DNN 2
                    ind_c_est, num_s = find_common_support(x_tildes_hat_b, thres=0.7)
                    accur_supp_2 = len(list(set(ind_c_est) & set(ind_c))) / max(len(ind_c), len(ind_c_est))

                    # common support est accuracy on training set for DNN 1
                    model1.eval()
                    accur_supp_1 = 1

                    # accuracy performance on VALIDATION set for DNN 1
                    r_tilde_val = val_dataset.r_tilde.detach()  # received signal
                    x_tilde_val = val_dataset.x_tilde.detach()  # true sparse channel
                    val_num = x_tilde_val.shape[0]

                    inputs2_val = torch.reshape(r_tilde_val, (-1, 2*K*M))
                    outputs1_val = model1(inputs2_val)
                    accur_supp_1_val = 1

                    # loss and accuracy performance on validation set for DNN 2
                    # get input for DNN2 combining data_batch2 and outputs1
                    outputs1_detach_val = outputs1_val.detach()  # dim = (num_val, N)
                    outputs1_detach_d_val = torch.repeat_interleave(outputs1_detach_val, K, dim=0)  # dim=(batch_size_1*K, N) check
                    inputs2_val = torch.cat((r_tilde_val, outputs1_detach_d_val), dim=1)  # dim=(batch_size_1*K, N + 2M)
                    r_val_hat, x_val_hat = model2(inputs2_val)
                    x_val_hat_real = x_val_hat[:, 0:N]
                    x_val_hat_imag = x_val_hat[:, N:]
                    loss_val = 2 * M * criterion2(r_val_hat, r_tilde_val) + \
                               gamma / val_num * torch.sum(torch.sqrt(x_val_hat_real ** 2 + x_val_hat_imag ** 2))
                    LOSS_val.append(loss_val.data.item())
                    # channel estimation performance
                    err_val = CE_err(x_tilde_val, x_val_hat)  # error in angular domain
                    ERR_val.append(err_val.data.item())

                    print('Epoch:', epoch, 'Itera:', itera, 'DNN1_step:', DNN1_step, 'Pretrain:', pretrain,
                          'loss2 =', '{:.6f}'.format(loss2.data.item()), 'loss2_val =', '{:.6f}'.format(loss_val.data.item()),
                          'err2 =', '{:.3f}'.format(err.data.item()), 'err2_val =', '{:.3f}'.format(err_val.data.item()),
                          'accur_1 =', '{:.3f}'.format(accur_supp_1), 'accur_2 =', '{:.3f}'.format(accur_supp_2),
                          'accur_1_val =', '{:.3f}'.format(accur_supp_1_val))
            itera = itera + 1

    return LOSS, ERR, LOSS_val, ERR_val


# Start experiment
print(torch.__version__)
t0 = time.time()  # start time from the beginning of the code

# set system parameters
N = 64  # BS tx antenna number
M = 50  # pilot length
K = 20  # number of users
rho = 1  # tx power
SNR = 30  # per channel SNR
fc = 30e9  # carrier frequency
c = 3e8  # speed of light
lamda = c / fc  # carrier wavelength
D = 0.5 * lamda  # antenna spacing

Pc = 2  # common scatter number
Pd = 1  # individual scatter number
P = Pc + Pd
kappa = 0.2

sys = {'N': N, 'M': M, 'K': K, 'rho': rho, 'SNR': SNR, 'fc': fc,
       'c': c, 'lamda': lamda, 'D': D, 'Pc': Pc, 'Pd': Pd}

# F is the unitary DFT matrix
F = DFTMatrix(N, f_type='DFT')  # the sparsifying basis of the channel, assuming linear array response

# obtain the pilot matrix (broadcasted to all the MUs)
A = generate_basis(N, M, rho, p_type='Gaussian')
# np.save('A.npy', A)
# A = np.load('A.npy')

S = A @ F.T.conj()
# S = np.load('S.npy')

num = 300000  # online training data available
val_num = 1000  # validation data available
test_num = 1000  # test data available

# create training data
train_dataset = MUMIMOData(sys, S, F, num)
A_tilde = train_dataset.get_A_tilde()
ind_set, ind_c = train_dataset.get_ind_set(), train_dataset.get_ind_c()
np.save('ind_set.npy', ind_set)
np.save('ind_c.npy', ind_c)

ind_set = np.load('ind_set.npy')
ind_c = np.load('ind_c.npy')
# train_dataset = MUMIMODataVT(sys, S, F, ind_set, num)
# A_tilde = train_dataset.get_A_tilde()
train_loader = DataLoader(dataset=train_dataset, batch_size=K, shuffle=False)

# create validation data
val_dataset = MUMIMODataVT(sys, S, F, ind_set, val_num)

# create the model1 for common support
in_size1 = 2 * M * K  # vectorized r from all users as input
out_size1 = N  # common support one hot vector
print('in_size1 =', in_size1, 'out_size1 =', out_size1)
widths1 = [in_size1, int(320), int(160), out_size1]
model1 = NetS(widths1, p=0)

# create the model2 for CE
in_size2 = 2 * M + N  # vectorized r_tilde and support probability (dim = N) as input
out_size2 = 2 * N  # vectorized h as input
print('in_size2 =', in_size2, 'out_size2 =', out_size2)
widths2 = [in_size2, int(320), int(320), out_size2]
model2 = Net(widths2, p=0)

# Set the learning rate and the optimizer for model 1
learning_rate1 = 0.001
optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate1, amsgrad=False)
# define BCE loss for training with labeled data
criterion1 = nn.BCELoss()

# Set the learning rate and the optimizer for model 1
learning_rate2 = 0.001
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate2, amsgrad=False)
# define MSE loss for training with labeled data
criterion2 = nn.MSELoss()

# train the model
# Set the model using dropout to training mode; this is the default mode, but it's good practice to write in code :
model1.train()
model2.train()
t1 = time.time()
LOSS, ERR, LOSS_val, ERR_val = train(model1, model2, criterion1, criterion2, train_loader, val_dataset,
                                     optimizer1, optimizer2, ind_c, epochs=1)
# in online training, we will have a infinite data set containing many real-time measurement and we won't loop over the
# dataset (i.e, there is only one epoch)
# compute training time
t2 = time.time()
elapsed = t2 - t1
print("Time used for training is", elapsed, 'seconds')

# save the model
torch.save(model1.state_dict(), 'model1.pt')
torch.save(model2.state_dict(), 'model2.pt')
# save the metrics during training as numpy array during training
np.save('LOSS_iter.npy', np.array(LOSS))
np.save('ERR_iter.npy', np.array(ERR))
np.save('LOSS_val_iter.npy', np.array(LOSS_val))
np.save('ERR_val_iter.npy', np.array(ERR_val))

# load the model
modeled1, modeled2 = NetS(widths1), Net(widths2)
modeled1.load_state_dict(torch.load('model1.pt'))
modeled2.load_state_dict(torch.load('model2.pt'))

# Set the model to evaluation mode, i.e., turn off dropout
modeled1.eval()
modeled2.eval()

# evaluate for SNR = 0:30 dB
SNRs = list(range(0, 31, 5))
SNR_num = len(SNRs)
ERR_test_SNR = []
ERR_clx_test_SNR = []
for s in range(SNR_num):
    sys['SNR'] = SNRs[s]
    test_data = MUMIMODataVT(sys, S, F, ind_set, test_num)
    r_test = test_data.r_tilde
    h_test = test_data.h_tilde
    x_test = test_data.x_tilde

    t11 = time.time()
    # forward pass for DNN 1
    inputs2_test = torch.reshape(r_test, (-1, 2 * K * M))
    outputs1_test = model1(inputs2_test)

    # input to DNN 2
    # get input for DNN2 combining data_batch2 and outputs1
    outputs1_detach_test = outputs1_test.detach()  # dim = (num_val, N)
    outputs1_detach_d_test = torch.repeat_interleave(outputs1_detach_test, K, dim=0)  # dim=(batch_size_1*K, N) check
    inputs2_test = torch.cat((r_test, outputs1_detach_d_test), dim=1)  # dim=(batch_size_1*K, N + 2M)
    _, x_hat_test = model2(inputs2_test)

    x_hat_test = x_hat_test.detach()
    t12 = time.time()
    err_test = CE_err(x_test, x_hat_test)

    h_test_np = h_test.numpy().T
    x_hat_test_np = x_hat_test.numpy().T
    h_test_clx = recover_complex_ch(h_test_np)      # N x num
    x_test_clx = recover_complex_ch(x_hat_test_np)     # N x num
    h_test_clx_hat = np.matmul(F, x_test_clx)
    err_clx_test = CE_clx_err(h_test_clx, h_test_clx_hat)

    ERR_test_SNR.append(err_test)   # real sparse channel error
    ERR_clx_test_SNR.append(err_clx_test)   # complex spatial channel error

np.save('err_test_SNR.npy', np.array(ERR_test_SNR))
np.save('err_clx_test_SNR.npy', np.array(ERR_clx_test_SNR))
print(ERR_clx_test_SNR, t12-t11)

# plot LOSS in training and validation
plt.figure()
plt.semilogy(LOSS, label='train loss')
plt.semilogy(LOSS_val, label='val loss')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('mean of loss function')
plt.grid()

# plot ERR in training and validation
plt.figure()
plt.plot(ERR, label='train err')
plt.plot(ERR_val, label='val err')
plt.xlabel('iteration')
plt.ylabel('MSE in dB')
plt.legend()
plt.grid()

# plot NMSE of channel estimation v.s. SNR
plt.figure()
plt.plot(SNRs, ERR_clx_test_SNR, '-*', label='Twostage Online DNN, pilot length = 30')

plt.xlabel('SNR in dB')
plt.ylabel('NMSE (dB) of channel estimation')
plt.legend()
plt.grid()

plt.show()
print('Finished by Henry')
