import torch
print(torch.cuda.is_available()) # 返回True则CUDA可用
print(torch.cuda.get_device_name(0)) # 返回第0个CUDA设备的名称，应该是您的3070

# import torch
# print(torch.__version__)