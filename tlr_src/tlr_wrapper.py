import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class TLR(Optimizer):
    def __init__(self, optimizer, meta_gamma=.20, meta_update=1./3, batches_per_epoch=100):

        self.optimizer = optimizer
        self.meta_update = int(meta_update * batches_per_epoch) # K = perc * N
        self.meta_gamma = meta_gamma

        self.cnt = 0

    @torch.no_grad()
    def step(self, closure=None):

        # perform typical optimizer step
        loss = self.optimizer.step(closure)

        # update every K epochs
        if (self.cnt == self.meta_update):
            self.cnt = 0
            self.update()

        self.cnt += 1

        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def update(self):

        for group in self.optimizer.param_groups:

            dlr, dlr2 = 0.0, 0.0
            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.optimizer.state[p]

                if 'cached_p' not in param_state:
                    param_state['cached_p'] = torch.clone(p.data) #.detach()

                if 'cached_buff' not in param_state:
                    param_state['cached_buff'] = torch.zeros_like(p.data)

                # old direction 
                u_old = torch.clone(param_state['cached_buff'])

                # new direction
                u_new = torch.clone(param_state['cached_p'] - p.data) / group["lr"]

                # store required auxiliary values
                param_state['cached_p'].copy_(p.data)
                param_state['cached_buff'].copy_(u_new)

                dlr += (u_old * u_new).sum().item()
                dlr2 += (u_old * u_old).sum().item()

            if dlr2 < 0:
                change = 1.0
            else:
                change = (1 + self.meta_gamma * dlr/ (dlr2 - dlr + 1e-10))
                #change = min(change, 100) 
           
            group['lr'] *= change 
            #print(dlr, dlr2, change, group['lr'])

    def get_lr(self):

        lrs = []
        for group in self.optimizer.param_groups:
            lrs += [group["lr"]]

        return lrs
