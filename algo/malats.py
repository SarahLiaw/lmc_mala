import torch
from torch.utils.data import DataLoader
from .base import _agent
from train_utils.dataset import sample_data
import math
import numpy as np


class MALATS(_agent):
    
    def __init__(self,
                 model,             # neural network model
                 optimizer,         # optimizer
                 criterion,         # loss function
                 collector,         # context and reward collector
                 eta=0.5,           # likelihood temperature
                 std_prior=1.0,     # Gaussian prior weight
                 beta_inv = 0.1,    # inverse temperature
                 accept_reject_step=0,
                 batch_size=None,   # batchsize to update nn
                 decay_step=20,     # learning rate decay step
                 reduce=None,       # reduce update frequency
                 device='cpu',
                 name='default'):
        super(MALATS, self).__init__(name)

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.collector = collector
        self.device = device

        # training-schedule params
        self.batchsize = batch_size
        self.decay_step = decay_step
        self.reduce = reduce
        self.base_lr = optimizer.lr
        self.step = 0

        # MALA hyperparameters
        self.eta = eta
        self.std_prior = std_prior
        self.beta_inv  = beta_inv
        self.accept_reject_step = accept_reject_step
        
        # replay buffer
        self.X, self.y = [], []
        self.loader = (DataLoader(collector, batch_size=batch_size) if batch_size else None)
        self.is_posterior_updated = True
        
        
    ############### Public-API hooks ############
    def clear(self):
        self.model.init_weights()
        self.collector.clear()
        self.step = 0
        self.X.clear()
        self.y.clear()
        self.is_posterior_updated = True

    @torch.no_grad()
    def choose_arm(self, context):
        pred = self.model(context)
        return int(torch.argmax(pred))

    def receive_reward(self, arm, context, reward):
        self.collector.collect_data(context, arm, reward)
        self.X.append(self.collector.last_feature)
        self.y.append(float(reward))
    #######################################
    
    def _potential_and_grad(self, ctx, r):
        self.model.zero_grad()
        preds = self.model(ctx).squeeze(1)
        data_term = self.criterion(preds, r)
        prior_term = sum((p**2).sum() for p in self.model.parameters())
        U = self.eta * data_term + self.std_prior * prior_term
        U.backward()
        grads = [p.grad.clone() for p in self.model.parameters()]
        return U.detach(), grads

    def _gd_step(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.model.parameters(), grads):
                p.add_(g, alpha=-lr)
    
    def _mala_proposal(self, lr, grads):
        noise_scale = math.sqrt(2 * lr * self.beta_inv)
        with torch.no_grad():
            for p, g in zip(self.model.parameters(), grads):
                noise = torch.randn_like(p) * noise_scale
                p.add_(-lr * g + noise)
                
    def _mala_move(self, lr, ctx, r, U_x, g_x):
        # save current params
        old_params = [p.data.clone() for p in self.model.parameters()]

        # propose
        self._mala_proposal(lr, g_x)
        U_y, g_y = self._potential_and_grad(ctx, r)

        # compute log‑proposal densities
        sq_xy = sq_yx = 0.0
        for p_new, p_old, g_new, g_old in zip(self.model.parameters(), old_params, g_y, g_x):
            sq_xy += (p_old - p_new + lr * g_new).pow(2).sum()
            sq_yx += (p_new - p_old + lr * g_old).pow(2).sum()
        log_q_xy = -sq_xy / (4 * lr * self.beta_inv)
        log_q_yx = -sq_yx / (4 * lr * self.beta_inv)
        log_alpha = -U_y + U_x + log_q_yx - log_q_xy

        # accept / reject
        if torch.log(torch.rand(1, device=self.device)) > log_alpha:
            # reject – restore parameters
            for p, old in zip(self.model.parameters(), old_params):
                p.data.copy_(old)
            return U_x, g_x          # keep old state
        return U_y, g_y              # accepted

    def _refresh_posterior(self, K):
        if not self.X:                                 
            return

        # build batch (same rule as LMCTS)
        if self.batchsize and len(self.X) > self.batchsize:
            idx = torch.randperm(len(self.X))[:self.batchsize]
            ctx = torch.stack([self.X[i] for i in idx]).to(self.device)
            r   = torch.tensor([self.y[i] for i in idx],
                               dtype=torch.float32, device=self.device)
        else:
            ctx = torch.stack(self.X).to(self.device)
            r   = torch.tensor(self.y, dtype=torch.float32, device=self.device)

        lr = self.base_lr / max(1, self.step)

        # initial potential & grad
        U, g = self._potential_and_grad(ctx, r)

        for _ in range(K):
            # optional GD warm‑up
            for _ in range(self.accept_reject):
                self._gd_step(lr, g)
                U, g = self._potential_and_grad(ctx, r)

            # one MALA move
            U, g = self._mala_move(lr, ctx, r, U, g)


    #####################################################
    def update_model(self, num_iter=5):
        self.step += 1
        if self.reduce and self.step % self.reduce:
            return

        if self.step % self.decay_step == 0:
            self.base_lr = self.base_lr / 2

        self.model.train()               
        self._refresh_posterior(num_iter)