# 名字：李志翔
# 创建时间:2021/8/6 10:44
#时间序列的表示方式
import torch.nn as nn
import torch

word_to_ix={"hello":0,"world":1}
lookup_tesor=torch.tensor([word_to_ix["hello"]],dtype=torch.long)

embeds=nn.Embedding(2,5)           #随机建表
hellon_embed=embeds(lookup_tesor)
print(hellon_embed)