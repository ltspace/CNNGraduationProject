from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random

import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import save_image
import time




# 检查文件夹是否存在，如果不存在，创建它
out_dir = './out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")  # 如果可用，使用CUDA
    print("当前正在使用的 GPU：", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")  # 如果不可用，回退到CPU
    print("CUDA不可用，正在使用CPU训练。")


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 设置一个随机种子，方便进行可重复性实验
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 数据集所在路径
dataroot = "C:/Users/14097/Desktop/cnn/data/Processed224x224data"

# 数据加载的进程数
workers = 0
# Batch size 大小 图片数量
batch_size = 16 ###
# Spatial size of training images. All images will be resized to this
# size using a transformer.
# 图片大小
image_size = 256

# nz是输入向量 z z z的长度

# nc是输出图像的通道数（对于RGB图像来说是3）
# 图片的通道数
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64 # ngf与通过生成器传播的特征图的大小有关
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 150
# Learning rate for optimizers
lr = 0.0001
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 3,512,512)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# We can use an image folder dataset the way we have it setup.
# Create the dataset 该类要求数据集的根文件夹中有子目录
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))


# plt.show()
# 权重初始化函数，为生成器和判别器模型初始化
def weights_init(m):
    classname = m.__class__.__name__
    # 卷积层权重
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
# 一系列的二维反卷积层完成创建与训练图像大小相同的图像（3x256x256)，每层都配带有批标准化层和relu激活
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入的是nz维度的噪声，想要生成一个 (ngf*8) x 4 x 4 的特征图
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 特征图大小 (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 特征图大小 (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 特征图大小 (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 特征图大小 (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # 特征图大小 (ngf//2) x 64 x 64
            nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            # 特征图大小 (ngf//4) x 128 x 128
            nn.ConvTranspose2d(ngf // 4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 最终输出大小 (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入尺寸 (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# 初始化生成器和判别器
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.2.
netG.apply(weights_init)
# Print the model
print(netG)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Initialize BCELoss function定义损失函数
criterion = nn.BCELoss() #使用二元交叉熵损失（BCELoss）函数

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 0.9
fake_label = 0.0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.1)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.1)
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    import time

    start = time.time()
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch

        if i % 2 == 0:
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            # view(-1)的作用是将 判别网络的输出数据维度变为一行与标签的维度相对应
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label) #计算损失函数
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item() #判别器对于真实批次的平均输出(整个批次) 理论上收敛到0.5

        ## Train with all-fake batch用所有的假数据训练
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G 用G生成假图片
        fake = netG(noise)

        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch 计算损失梯度
        errD_fake.backward()
        D_G_z1 = output.mean().item() #判别器对于假批次的平均输出。第一个数字在 D D D更新之前
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # 判别器损失，计算为所有真实批次和所有假批次的损失之和(log(D(x)) + log(D(G(z)))
        # Update D 更新D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label) #生成器损失，计算为log(D(G(z)))
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()# 判别器对于假批次的平均输出 在 D D D更新之后 数字在开始的时候应该是接近0的，并随着 G G G的提高向0.5收敛

        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 20 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            #   #make_grid的作用是将若干幅图像拼成一幅图像。
            #   # 其中padding的作用就是子图像与子图像之间的pad有多宽。
            #   #此处将64张图像拼接为一张图像间隔为0pixel
            img_list.append(vutils.make_grid(fake, padding=0, normalize=True))###

            # i = vutils.make_grid(fake, padding=2, normalize=True)
            # fig = plt.figure(figsize=(8, 8))
            # plt.imshow(np.transpose(i, (1, 2, 0)))
            # plt.axis('off')  # 关闭坐标轴
            # plt.savefig("out/%d_%d.png" % (epoch, iters))
            # plt.close(fig)
        iters += 1
        # 存储生成的图片
        # 存储的是一个64*64*64的图像，每一张图像之间是0像素
        # print(img_list[len(img_list)-1].shape)
        real_images = to_img(img_list[len(img_list) - 1].cpu().data)
        name = './out/' + str(len(img_list) - 1) + '.jpg'
        save_image(real_images, name)
        # print(type(real_images))
        # 保存最后一张图像单张图像
        if epoch == num_epochs - 1:
            fullimage = cv2.imread(name)
            h, w = fullimage.shape[:2]
            for i in range(int(h / 64)):
                for j in range(int(w / 64)):
                    partimg = fullimage[64 * i:64 * i + 64, 64 * j:64 * j + 64]
                    name = './out/img' + str(i) + str(j) + '.jpg'
                    cv2.imwrite(name, partimg)
    print('time:', time.time() - start)

    # 更新学习率
    schedulerD.step()
    schedulerG.step()

# Grab a batch of real images from the dataloader
# real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
