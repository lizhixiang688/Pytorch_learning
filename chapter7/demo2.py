# 名字：李志翔
# 创建时间:2021/8/1 17:29
#argmax
import torch
from torch.nn import functional as F

logits=torch.rand(4,10)
pred=F.softmax(logits,dim=1)
pred_label=pred.argmax(dim=1)
logits_label=logits.argmax(dim=1)
label=torch.tensor([9,3,2,4])
correct=torch.eq(pred_label,label)
print(correct.sum().float().item()/4)
