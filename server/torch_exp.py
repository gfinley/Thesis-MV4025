import torchsummary
from torchsummary import summary
import torch
#load the model

model =  torch.jit.load("ai/model.pt")

sub_models = model.torch_sub_mod
print(model.torch_sub_model)

print(model.torch_sub_model.code)

#summary(model.torch_sub_model, (12,6,6))