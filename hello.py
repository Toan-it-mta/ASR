print('Hello Slurm!')
import torch

print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
print("torch.cuda.device:", torch.cuda.device(0))
print("torch.cuda.current_device:", torch.cuda.current_device())
print("List torch.device:", [ torch.device("cuda:%s" % i) for i in range(torch.cuda.device_count())])
print("List torch.cuda.device:", [torch.cuda.device(i) for i in range(torch.cuda.device_count())])
print("torch.cuda.get_device_name:", torch.cuda.get_device_name(0))