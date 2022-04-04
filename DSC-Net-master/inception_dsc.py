import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering import spectral_clustering, acc, nmi
import scipy.io as sio
import math
from torch.autograd import Variable
from collections import OrderedDict

"""用pytorch实现TensorFlow的SamePad"""

class Conv2dSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)

"""用pytorch实现TensorFlow的Same逆卷积"""

class ConvTranspose2dSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwarg):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwarg)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwarg):
        super(BasicConvTranspose2d, self).__init__()
        self.convtranspose = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwarg)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.convtranspose(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1_pad = Conv2dSamePad(kernel_size=1,stride=2)   
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1,stride=2)

        self.branch3x3_1 = BasicConv2d(in_channels, 16, kernel_size=1)
        self.branch_pad = Conv2dSamePad(kernel_size=3,stride=2)
        self.branch3x3_2 = BasicConv2d(16, 64, kernel_size=3,stride=2)

        self.branch5x5dbl_1 = BasicConv2d(in_channels, 16, kernel_size=1)
        self.branch5x5dbl_2 = BasicConv2d(16, 64, kernel_size=3)
        self.branch5x5dbl_3 = BasicConv2d(64, 64, kernel_size=3,stride=2)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1_pad(x)
        branch1x1 = self.branch1x1(branch1x1)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch_pad(branch3x3)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5dbl = self.branch5x5dbl_1(x)
        branch5x5dbl = self.branch_pad(branch5x5dbl)
        branch5x5dbl = self.branch5x5dbl_2(branch5x5dbl)
        branch5x5dbl = self.branch_pad(branch5x5dbl)
        branch5x5dbl = self.branch5x5dbl_3(branch5x5dbl)

        branch_pool = self.branch_pad(x)
        branch_pool = F.avg_pool2d(branch_pool, kernel_size=3, stride=2)
        branch_pool = self.branch_pool(branch_pool)

        y = [branch1x1, branch3x3, branch5x5dbl, branch_pool]

        y = torch.cat(y, 1)
        return y


class InceptionB(nn.Module):

    def __init__(self, in_channels,branch_channels):
        super(InceptionB, self).__init__()
        # self.branch1x1_pad = ConvTranspose2dSamePad(kernel_size=1,stride=2)
        # self.branch1x1 = BasicConvTranspose2d(in_channels, branch_channels, kernel_size=1,stride=2)

        self.branch5x5_pad = ConvTranspose2dSamePad(kernel_size=5,stride=2)
        self.branch5x5 = BasicConvTranspose2d(in_channels, branch_channels, kernel_size=5,stride=2)

        self.branch3x3_pad = ConvTranspose2dSamePad(kernel_size=3,stride=2)
        self.branch3x3 = BasicConvTranspose2d(in_channels, branch_channels, kernel_size=3,stride=2)



    def forward(self, x):

        # branch1x1 = self.branch1x1(x)
        # print(branch1x1.shape)
        # branch1x1 = self.branch1x1_pad(branch1x1)

        branch5x5 = self.branch5x5(x)
        branch5x5 = self.branch5x5_pad(branch5x5)


        branch3x3 = self.branch3x3(x)
        branch3x3 = self.branch3x3_pad(branch3x3)


        y = [branch3x3, branch5x5, branch3x3]
        y = torch.cat(y, 1)
        return y

"""不含自表达层的自编码器"""

class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        inception_blocks = [
            BasicConv2d, InceptionA,InceptionB,BasicConvTranspose2d
        ]
        assert len(inception_blocks) == 4
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        convtranspose_block = inception_blocks[3]

       
        self.convpad_3x3 = Conv2dSamePad(kernel_size=3,stride=2)
        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        self.Mixed_1b = InceptionA(32, pool_features=32)
        self.Mixed_1c = InceptionA(224, pool_features=64)
        
        self.Mixed_2a = InceptionB(256,64)
        self.Mixed_2b = InceptionB(192,32)
        self.Convtranspose2d_2b_3x3 = BasicConvTranspose2d(96,1,kernel_size=3,stride=2)
        self.convtranspad_3x3 = ConvTranspose2dSamePad(kernel_size=3, stride=2)
    def forward(self, x):
        x = self.convpad_3x3(x)
        x = self.Conv2d_1a_3x3(x)
        x = self.Mixed_1b(x)
        x = self.Mixed_1c(x)

        x = self.Mixed_2a(x)
        x = self.Mixed_2b(x)
        x = self.Convtranspose2d_2b_3x3(x)
        x = self.convtranspad_3x3(x)
        return x

class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y

class DSCNet(nn.Module):
    def __init__(self, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE()
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp

        return loss

"""预训练"""

def pre_train(model,  # type: ConvAE
          x, y, epochs, lr=1e-3, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8, show=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    criterion = torch.nn.MSELoss(reduction='sum')
    for epoch in range(epochs):
        x_recon = model(x)
        loss = criterion(x_recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:            
            print("Loss is:{:.4f}".format(loss.item()))

def train(model,  # type: DSCNet
          x, y, epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8, show=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    for epoch in range(epochs):
        x_recon, z, z_recon = model(x)
        loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        optimizer.zero_grad()
        loss
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' %
                  (epoch, loss.item() / y_pred.shape[0], acc(y, y_pred), nmi(y, y_pred)))

"""进行微调得到自表达层"""

if __name__ == "__main__":
    import argparse
    import warnings
    names = locals()
    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='orl',
                        choices=['coil20', 'coil100', 'orl', 'reuters10k', 'stl'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('-f', type=str, default="读取jupyter的额外参数")
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db = args.db
    if db == 'coil20':
        # load data
        data = sio.loadmat('datasets/COIL20.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        K = len(np.unique(y))
        #数据预处理，图像分块
        image = x
        image = image.reshape(-1,2,16,32)
        image = np.transpose(image,(1,0,2,3))
        for l in range(1, 3):
          names['x' + str(l)] = image[l - 1].reshape(-1,1,16,32)       
          names['num_sample' + str(l)] = names['x' + str(l)].shape[0]
        # network and optimization parameters
        channels = [1, 15]
        kernels = [3]
        epochs = 40
        weight_coef = 1.0
        weight_selfExp = 75

        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
        warnings.warn("You can uncomment line#64 in post_clustering.py to get better result for this dataset!")
    elif db == 'coil100':
        # load data
        data = sio.loadmat('datasets/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 50]
        kernels = [5]
        epochs = 120
        weight_coef = 1.0
        weight_selfExp = 15

        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
    elif db == 'orl':
               # load data
        data = sio.loadmat('datasets/ORL_32x32.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        print(x)
        # network and optimization parameters
        num_sample = x.shape[0]
        epochs = 700
        weight_coef = 2.0
        weight_selfExp = 0.2

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    
    x = torch.tensor(x, dtype=torch.float32, device=device)
#     预训练
    convae = ConvAE()
    convae.to(device)
    pre_train(convae, x, y, 5000, alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=args.show_freq, device=device)
    torch.save(convae.state_dict(), args.save_dir + '/%s.pkl' % args.db)
    
    
    
    #微调
    C = nn.Parameter(1.0e-8 * torch.zeros( num_sample,  num_sample, dtype=torch.float32), requires_grad=True).detach().to('cpu').numpy()

    dscnet = DSCNet(num_sample=num_sample)
    dscnet.to(device)
    
    convae_state_dict = torch.load(args.save_dir + '/%s.pkl' % args.db )
    dscnet.ae.load_state_dict(convae_state_dict)
    print("Pretrained ae weights are loaded successfully.")

    train(dscnet, x, y, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
          alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=args.show_freq, device=device)
    torch.save(dscnet.state_dict(), args.save_dir + '/%s-model.ckp' % args.db)