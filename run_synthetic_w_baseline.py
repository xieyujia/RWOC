#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:21:29 2020

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:57:39 2020

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:22:25 2020

@author: yujia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:07:08 2020

@author: yujia
"""

import argparse
import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn import Module
import numpy as np
import torch.nn.functional as F
from sinkhorn import Sinkhorn_custom
from snn import SNet

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 



parser = argparse.ArgumentParser(description='Process some integers.')
# data generation
parser.add_argument('--data_type', type=str, default='uniform',
                    help='normal | mix_normal | uniform')
parser.add_argument('--n', type=int, default=int(100),
                    help='total number of data')
parser.add_argument('--train_level', type=float, default=1.,
                    help='proportion of the total data used for training')
parser.add_argument('--d1', type=int, default=1,
                    help='dimension of the features on the first platform')
parser.add_argument('--d2', type=int, default=1,
                    help='dimension of the features on the second platform')
parser.add_argument('--noise_level', type=float, default=0.0,
                    help='noise added')
parser.add_argument('--seed_data', type=int, default=1,
                    help='seed for data generation')

# training
parser.add_argument('--train_iter', type=int, default=500,
                    help='total number of traning steps')
parser.add_argument('--lr_R', type=float, default=1e-3,
                    help='learning rate for regression')
parser.add_argument('--seed_train', type=int, default=0,
                    help='seed for traning')

# print
parser.add_argument('--print_every', type=int, default=1,
                    help='number of epochs for each print')
parser.add_argument('--visual', type=int, default=1,
                    help='whether visual is wanted')
parser.add_argument('--save_val_result', type=str, default=None,
                    help='path to store the validation result')

args = parser.parse_args()


np.random.seed(args.seed_data)
torch.manual_seed(args.seed_train)


n = args.n
d1 = args.d1
d2 = args.d2
noise = args.noise_level
bs = n
max_iter = args.train_iter

lr_R = args.lr_R
print_every = args.print_every

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.data_type == 'normal':
    x = np.random.normal(0,1,size=[n, d1+d2])
elif args.data_type == 'mix_normal':
    x = np.concatenate([np.random.normal(-1,1,size=[int(n/2), d1+d2]), 
                        np.random.normal(2,4,size=[int(n/2), d1+d2])], axis=0)
elif args.data_type == 'uniform':
    x = np.random.uniform(0,1,size=[n, d1+d2])


w = np.random.normal(size=[d1+d2,1])
y = x.dot(w) + noise*np.random.normal(size=[n,1])

w_for_print = np.squeeze(w)


x = torch.FloatTensor(x).to(device)
y = torch.FloatTensor(y).to(device).squeeze(-1)

    
Rmodel = torch.nn.Linear(d1+d2, 1, bias=False).to(device)


optimizer_R = torch.optim.SGD(
    Rmodel.parameters(), lr=lr_R, momentum=0.9, weight_decay=5e-4)


loss_list = []
epoch_count = 0
for batch_idx in range(max_iter):
#    print(batch_idx)
    # Get data
    optimizer_R.zero_grad()
    pred = Rmodel(x).squeeze(-1)

    loss = torch.mean((y-pred)**2)
    loss.backward()
    optimizer_R.step()

    loss_list.append(loss.item())
#    print('loss:', batch_loss_list[-1])
    
    epoch_count += 1
        
    if epoch_count % print_every == 0:
        print('epoch:', epoch_count, 'loss:', loss_list[-1])
        
        print(w_for_print)
        print(list(Rmodel.parameters())[0].data.numpy())
        print('-'*40)
    
    
#    print(loss, list(Rmodel.parameters()))

#%%
plt.plot(loss_list)
plt.show()
        #%%
    
print(w_for_print)
print(list(Rmodel.parameters())[0].data)

