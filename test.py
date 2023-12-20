import torch
import torch.nn as nn

emb = nn.Embedding(5,100)
index1 = torch.tensor(4)
index2 = torch.tensor(5)
linear = nn.Linear(100,5)
try:
    emb(index2)
except Exception as e:
    print(e)

emb = emb.to('cuda')
index1 = index1.to('cuda')
index2 = index2.to('cuda')
linear = linear.to('cuda')
try:
    test_data = emb(index1)
    a = emb(index2)
    test_data = linear(test_data)
    print(1)
except Exception as e:
    print(e)
