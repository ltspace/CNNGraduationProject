### 3/20 19点50分，开干

打算用VGG微调

先把我用ps抠出来的的一个个透明背景的工件背景设为黑色

前几天用vott标注了一下毛刺，但这次打算不整了

rgba转rgb

然后把不规则的图片图形用黑色填充为正方形

想用GAN生成点儿数据

cpu训太慢了，想用3070训，装CUDA，装完加环境变量里重启

又发现自己的pytorch是cpu版的，和cuda也不配，又去换

跑起来了  但是训练到到第二轮就开始模式崩溃

gan的代码是64size直接改的，只改了size和卷积层，怀疑是因为没有调整其他超参数



--凌晨一点电脑没电，睡觉--

### 3/21 7点57分，开干

lr从0.0003降低到0.0001，学的慢，但依然模式崩溃

怀疑是自己给的是不同的工件的图像，只留下一种训练，依然模式崩溃

为了平衡，减少判别器更新，依然模式崩溃

调整真实样本的标签值real_label = 0.9，依然模式崩溃

转用imgaug和albumentations数据增强

imagaug没问题，试用album的时候生成的图像全黑

删去normalize的部分，生成正常

毛刺正好在图像的最边缘，不能裁切

图片名称规格化

改变心意，想用ResNet

数据集划分想用k折交叉验证

256->224，发现生成了全是名字为1.png之类的文件夹，发现是目录结构得在上一层，不能直接放在文件夹里边，还得新建个文件夹嵌套一下，有的需要这样有的不需要

一共75张，选层数少的resnet18或者34，防止过拟合

我现在一听Am I Dreaming就厌世

脚本创建CSV填好名称，打标签，化身数据标注女工

0无1有

开始训练，100epoch

太多了，改成50



![image-20240321160517361](C:\Users\14097\AppData\Roaming\Typora\typora-user-images\image-20240321160517361.png)

![image-20240321162341172](C:\Users\14097\AppData\Roaming\Typora\typora-user-images\image-20240321162341172.png)

![image-20240321170513055](C:\Users\14097\AppData\Roaming\Typora\typora-user-images\image-20240321170513055.png)



resnet结构

1. 导入必要的库：代码一开始导入了所需的库，包括 PyTorch 相关的模块、Pandas 用于数据处理、PIL 用于图像处理等。
2. 加载预训练的 ResNet-18 模型并微调：通过加载预训练的 ResNet-18 模型，并将其全连接层修改为适合二分类任务的形式。然后将模型设置为训练模式。
3. 自定义数据集类 CustomDataset：定义了一个自定义的数据集类，用于加载图像数据和对应的标签，并在需要时应用数据转换。
4. 图像预处理和数据增强：定义了图像预处理和数据增强的操作，包括大小调整、转换为张量、归一化等。
5. 训练集和验证集划分：将数据集划分为训练集和验证集，用于模型训练和评估。
6. 定义损失函数和优化器：定义交叉熵损失函数和 Adam 优化器。
7. 训练和验证循环：循环训练模型并在验证集上评估模型性能，输出每个 epoch 的训练损失和验证损失。
8. 测试模型性能：使用测试数据集评估模型在未见过的数据上的性能，计算模型的准确率。

ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=2, bias=True)
)



[0/150][0/4]    Loss_D: 3.0506  Loss_G: 28.2704 D(x): 0.2642    D(G(z)): 0.3776 / 0.0000
time: 9.075353145599365
[1/150][0/4]    Loss_D: 0.0000  Loss_G: 69.0810 D(x): 1.0000    D(G(z)): 0.0000 / 0.0000
time: 8.23465347290039

Epoch 9, Train Loss: 0.09285994420448938, Validation Loss: 2.735260248184204
Epoch 10, Train Loss: 0.03640504243473212, Validation Loss: 1.4351848363876343
Accuracy of the model on test images: 92.0%



- 一种常见的做法是应用**判别器的标签平滑（label smoothing）**，即在训练判别器时不使用硬标签（0和1），而是使用一些接近0和1的软标签。
- 另一种方法是使用不同的损失函数，例如Wasserstein损失，它可以提供更稳定的梯度。
- 对抗性训练本身是一个敏感且有时不稳定的过程。因此，您可能需要**调整学习率**，或者是**增加生成器相对于判别器的训练频率**。
- 使用梯度惩罚（gradient penalty）或者实例噪音（instance noise）等技术来稳定训练过程。
- 检查生成器和判别器的网络构架是否合理，以及是否有梯度消失或梯度爆炸的问题发生。
- 适时改变生成器的输入噪声向量，防止生成器陷入产生过于相似输出的局部最优解。





您提供的代码段是一个判别器模型的定义，用于神经网络中的一种生成对抗网络(GAN)架构。该模型使用多层卷积网络(Convolutional Neural Network, CNN)，搭配批归一化（BatchNorm）和Leaky ReLU激活函数构建。模型的目的通常是判断输入数据是真实数据还是生成器生成的假数据。



这个判别器由多个卷积层组成，每一层都有其对应的卷积核数量（例如，`ndf`、`ndf * 2`、`ndf * 4`、`ndf * 8`），卷积步长和padding设置保证了数据维度的减半处理，批归一化帮助模型快速收敛并减少内部协变量偏移，LeakyReLU则提供非线性处理，最后通过一个卷积层和Sigmoid函数输出一个值，用于表征数据真实与否的概率。



如果您想使用这个判别器来检测图像中的毛刺（假设“毛刺”是某种可区分特征），您可能需要对训练数据进行处理，让模型学习这些特征代表假数据。模型训练后，可以将图像输入判别器，观察输出概率值，高概率值可能表明图像中含有毛刺。



请注意，为了使判别器工作，您需要设定几个参数，比如`nc`（输入数据的通道数），`ndf`（模型的特征图数量），以及`ngpu`（用于训练的GPU数量）。此外，您还要准备相应的训练和测试数据集，并对生成对抗网络进行训练，这通常需要一定量的正面样本（无毛刺的图像）和负面样本（有毛刺的图像）。如果您需要更具体的引导或帮助，请提供更多信息或问我具体问题。





如果你打算使用卷积神经网络（CNN）模型来解决你的毕设工件毛刺图像的二分类问题，并且是通过微调大模型来进行，我有以下几点建议：



1. **选择合适的预训练模型**：你可以选择在类似数据集上表现良好的预训练模型。例如，ResNet、VGGNet、Inception等模型在图像分类任务中表现优异，可以作为起点。
2. **数据预处理**：在训练之前，确保对数据集进行恰当的预处理，如缩放、归一化和数据增强等，以提高模型的泛化能力。
3. **迁移学习策略**：由于你的数据集可能规模较小，你应该冻结预训练模型的大部分层，仅对顶层进行微调。这样可以借用预训练模型学到的特征，同时避免模型在小数据集上过拟合。
4. **微调细节**：你可以替换预训练模型的分类层，以匹配你的具体二分类问题。确保学习率适当，对于冻结的层可以设置为0，而微调层可以设置稍高的学习率。
5. **损失函数和评价指标**：由于是二分类问题，通常采用二元交叉熵作为损失函数。同时，精确度、召回率、F1得分和ROC曲线是评估二分类任务的常用指标。
6. **正则化和优化器选择**：使用适当的正则化技术，如Dropout或权重衰减，以及适当的优化器，如Adam，以改善训练过程。
7. **模型训练和验证**：使用交叉验证等方法来监控模型在独立数据集上的表现，并调整超参数以获取最佳性能。
8. **模型解释性**：可视化卷积层的特征图可以帮助你理解模型是如何识别毛刺的，这对于后续的模型迭代和改进是有益的。



在开始之前，确保你有一个清晰的工作计划，明确微调哪些层，以及设置恰当的实验进行比较。





使用VGGNet进行微调是一个很好的选择，因为其架构简单且在图像识别任务中经常被作为基线模型。以下是使用VGGNet进行微调解决工件毛刺图像分类问题的步骤：



1. **预训练模型的选择**：选择VGGNet的一个预训练版本。VGG16和VGG19是两个常用的变体。这些模型在ImageNet数据集上预训练，能够捕捉丰富的图像特征。
2. **数据预处理**：调整你的图像数据大小以匹配VGGNet的输入要求（通常为224×224像素），并应用与预训练模型相同的归一化方法。
3. **模型修改**：移除原始VGGNet模型的全连接层，并替换为适合你二分类问题的新层。例如，你可以添加一个带有两个输出的全连接层，并使用sigmoid激活函数。
4. **训练策略**：冻结除了顶层之外的所有卷积层，因为初期的许多特征（如边缘和纹理）在多数视觉任务中都是通用的。只训练你刚添加的全连接层几个epoch。
5. **微调**：在新层有些许训练之后，你可以开始解冻一些卷积层，并对整个网络进行细微调整。解冻较深层次的卷积层，逐步降低学习率，继续训练。
6. **超参数调整**：经验表明，较小的学习率（如1e-4或1e-5）对于微调预训练网络是很有用的。同时，监控验证集的性能指标，避免过拟合。
7. **训练监控**：使用早期停止（early stopping）或者模型检查点来监控模型性能，确保你能得到最好的模型，并能在过拟合发生时停止训练。



祝你使用VGGNet微调的工作进展顺利。记得在实验过程中记录每一步的变化和结果，这样有助于理解哪些改进对模型的性能提升最有效。





你提供的是一个基于生成对抗网络（GAN）的Python代码，用于生成新的图像数据。这个代码主要包括两个部分：一个生成器(Generator)和一个判别器(Discriminator)。生成器用于生成新的图像，而判别器用于评估这些图像是否足够真实。数据增强通常是指在现有数据集上应用变换来扩充数据集的过程。



要使用这个代码进行数据增强来生成毛刺图像，你需要做的是：



1. **训练GAN**：使用你的工件图像数据集训练这个GAN网络。训练过程中，生成器试图生成含有毛刺的新工件图像（如果这是你要求的），而判别器则试图区分真实图像和由生成器生成的图像。
2. **生成图像**：训练完成后，你就可以使用生成器生成批量新的含毛刺的工件图像。你可以通过向生成器提供不同的输入噪声向量来生成不同的图像，这些噪声向量应该有与你定义的`nz`（潜在向量z的大小）相同的长度。
3. **保存图像**：生成的新图像可以使用`save_image()`函数保存到你指定的路径。



要开始生成数据，请确保：



- `dataroot`变量正确地设置为含有你毕设工件图像的文件夹路径。
- 相关的图像预处理转换与你的图像数据集匹配。例如，如果你的工件图像尺寸不是64x64，你需要调整`transforms.Resize()`和`transforms.CenterCrop()`方法中的参数。



现有代码在生成器（Generator）的末端使用了`Tanh`激活函数，这将生成的图像像素值限制在 [-1, 1] 范围内。因此，在使用生成的图像前，你可能需要取消归一化（可以使用`to_img()`），将像素值转换回 [0, 1] 范围。这部分代码已经包含在提供的代码中。



此外，如果你想产生特定类型的变化（例如旋转、放缩、颜色变换等），你可以在数据加载器中加入更多的变换操作。



请注意，GAN是一个强大但复杂的工具，训练过程可能需要一些调整来获得优质的生成图像。如果你之前没有使用过GAN，可能需要一些实验来调整网络架构、训练策略和超参数。此外，生成的图像质量很大程度上取决于GAN的训练程度和原始数据集的多样性。

工件图像尺寸不是64x64，是长条形状的，例如111*486，而且每个图片尺寸都不一样

如果工件图像具有不同的尺寸，并且它们都是长条形状，例如111x486，并且每个图像的尺寸都不同，你需要在预处理步骤中做一些调整。这里有几种处理不同尺寸图像的方法：



1. **缩放图像**：
   你可以选择将所有图像缩放到统一的尺寸。这意味着所有图像都将被变形以适应新的尺寸。这可能会导致一些图像失真，特别是那些尺寸和宽高比例差异很大的图像。
2. **填充图像**：
   另一种方法是通过填充（padding）来处理图像，这样可以保持原始宽高比，同时使所有图像具有相同的尺寸。填充可以添加到图像的宽度或高度，以匹配目标尺寸。
3. **裁剪图像**：
   如果你的图像中有共同的感兴趣区域（ROI），你也可以选择裁剪图像，使它们具有相同的尺寸。这种方法有可能丢失一些信息，所以需谨慎使用。



锁的类型，变量啥的存储在内存中的什么区域，还有一些C++的基础知识(这个问了好多我都没答出来，现在记忆模糊)，tcp和udp区别，进程切换的准备工作，进程与线程那一堆，tcp建立连接的过程



我是把工件都抠图抠出来了，背景是透明的，这种情况该怎么办，填充黑色会不会有些奇怪，是不是应该直接把透明的地方也改成黑色的？透明的部分rgb是什么 我很好奇

在您的场景中，如果工件的背景是透明的，并且会进行透明背景去除（抠图），在进行图像处理时，您确实需要决定如何处理透明部分。常见的处理方法包括：



1. **填充固定颜色**：您可以选择将背景填充为固定颜色，如黑色或白色。 在机器学习中，通常使用黑色填充（RGB值为(0, 0, 0)），因为黑色的像素值为零，在某些情况下可能有助于网络学习。
2. **使用均值填充**：在某些复杂的应用场景中，可能会选择用图像的均值填充背景，以尽可能减少背景对模型识别效果的影响。



透明部分在图像文件（如PNG格式）中由Alpha通道控制，Alpha值定义了像素的透明度，范围从0（完全透明）到255（完全不透明）。RGB和Alpha一起时常表示为RGBA，对于完全透明的像素，RGBA的值通常是(0, 0, 0, 0)，这里RGB值通常是不被渲染的，因为完全透明意味着在这些像素上看不到颜色。



对于机器学习或计算机视觉任务，透明背景对分类或检测任务通常是不需要的，因此您可以将透明部分处理为单一的背景颜色以避免混淆模型。 如果您选择填充背景色，您可以使用之前提供的代码范例，但使用黑色填充，因为神经网络通常能较好地从黑色背景中区分出前景物体。



处理带有透明背景的图像代码示例可以参考如下：



python

```python
from PIL import Image, ImageOps

class TransparentImageToTensor:
    def __call__(self, image):
        # 假定输入的图片为带透明度的RGBA图像
        # 将透明度通道转换为白色/黑色背景，在这里选择黑色
        rgb_image = image.convert('RGB')
        return rgb_image

transform = transforms.Compose([
    TransparentImageToTensor(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 其他创建数据集和加载器的代码保持不变
```



请注意调整这些处理步骤以适应你的具体需求。由于您的背景已经是透明的，处理起来相对简单；如果要求更高级别的背景处理（例如上下文填充、模式填充等），可能还需要一些高级图像处理技术。







```python
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
```



这行代码检查了CUDA是否可用，它是NVIDIA GPU上运行深度学习任务的计算平台和API模型。如果CUDA可用且`ngpu`（表示您希望用于训练的GPU数量）大于0，则`device`将被设置为 `"cuda:0"`，表示训练将在第一个可用的GPU上进行。如果CUDA不可用或`ngpu`等于0，则`device`将被设置为 `"cpu"`，表示训练将在CPU上进行。



除了上述设置外，为了实现在GPU上训练，您需要确保模型和数据都发送到了GPU上。在您的代码中，有关模型到GPU的代码是：



```python
# Create the generator
netG = Generator(ngpu).to(device)
# Create the Discriminator
netD = Discriminator(ngpu).to(device)
```