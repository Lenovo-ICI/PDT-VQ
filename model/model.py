import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpq import DPQ
from .drq import DRQ
from .qinco import QINCo
from .trans_layers import DefaultTransLayer, OrthogonalTrans, MLPTrans, MultiStepDistributionTrans
from utils.loss import distance_loss


class PDT(nn.Module):
    def __init__(self, d, args) -> None:
        super().__init__()
        """
        d: original vector dimension
        d_hidden: transformed vector dimension
        M: number of codes
        K: bits of each code
        """
        self.args = args
        self.M = args.M
        self.K = args.K
        self.d, self.d_hidden = d, args.d_hidden
        self.step_norm = not args.no_step_norm
        self.head_norm = not args.no_head_norm
        self.ms_sup = args.ms_sup
        
        if self.args.trans_type == 'no':
            self.trans = DefaultTransLayer(self.d)
        elif self.args.trans_type == 'orth':
            self.trans = OrthogonalTrans(self.d)
        elif self.args.trans_type == 'mlp':
            self.trans = MLPTrans(self.d, self.d_hidden, args.steps)
        elif self.args.trans_type == 'msd':
            self.trans = MultiStepDistributionTrans(self.d, self.d_hidden, self.M, args.steps, args.heads, self.step_norm, self.head_norm)
        else:
            raise NotImplementedError

        if self.args.vq_type == 'dpq':
            print('using deep product quantizer')
            self.vq = DPQ(self.d, self.M, self.K, args.codebook_init)
        elif self.args.vq_type == 'drq':
            print('using deep residual quantizer')
            self.vq = DRQ(self.d, self.M, self.K, args.codebook_init)
        elif self.args.vq_type == 'qinco':
            print('using qinco quantizer')
            self.vq = QINCo(self.d, self.M, self.K, args.qinco_h, args.qinco_L, args.codebook_init)
        else:
            raise NotImplementedError
    
    def init_codebook(self, x, resume=None):
        x = self.encode(x)
        self.vq.init_codebook(x, resume)

    def init_transform(self, resume):
        if resume is not None:
            state_dict = torch.load(resume)
            new_state_dict = {}
            for k, v in self.state_dict().items():
                if 'trans' in k:
                    new_state_dict[k] = state_dict[k]
                else:
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict)
    
    def encode(self, x, out_step=None):
        return self.trans.encode(x, out_step)
    
    def get_codes(self, x):
        return self.vq.get_codes(x)
    
    def reconstruction(self, codes):
        return self.vq.reconstruction(codes)
    
    def decode(self, x, out_step=None):
        return self.trans.decode(x, out_step)
    
    def forward_test(self, x):
        x_enc = self.encode(x, out_step=self.args.steps)[-1]
        x_recon_vq, _, side_output = self.vq(x_enc)
        x_recon = self.decode(x_recon_vq, out_step=self.args.steps)[-1]
        loss = torch.norm(x_recon - x, 2, dim=-1) + distance_loss(x_recon, x)
        return loss
    
    def forward_train(self, x):
        if self.args.vq_type != 'qinco':
            if not self.ms_sup:
                x_enc = self.encode(x, out_step=self.args.steps)[-1]
                x_recon_vq, _, side_output = self.vq(x_enc)
                x_recon = self.decode(x_recon_vq, out_step=self.args.steps)[-1]
                loss = torch.norm(x_recon - x, 2, dim=-1) + distance_loss(x_recon, x)
                return loss
            else:
                tau = 2
                norm = torch.exp(torch.as_tensor([i / tau for i in range(self.args.steps)])).sum().to(x.device)
                xs_enc = self.encode(x, out_step=self.args.steps)
                x_recons = []
                side_outputs = []
                for i in range(self.args.steps):
                    x_recon_vq_i, _, side_output_i = self.vq(xs_enc[i])
                    x_recon_i = self.decode(x_recon_vq_i, out_step=i+1)[-1]
                    x_recons.append(x_recon_i)
                    side_outputs.append(side_output_i)
                loss = torch.zeros(len(x)).to(x.device)
                for k, x_recon in enumerate(x_recons):
                    weight = torch.exp(torch.tensor(k/tau).to(x.device))
                    loss += weight / norm * (torch.norm(x_recon - x, 2, dim=-1) + distance_loss(x_recon, x))
        else:
            x_enc = self.encode(x)[-1]
            x_recon_vq, _, side_output = self.vq(x_enc)
            x_recon = self.decode(x_recon_vq)
            loss = [torch.norm(out - x_enc, 2, dim=-1) for out in side_output]
            loss = torch.stack(loss).mean(dim=0)
            if self.args.trans_type != 'no':
                loss_trans = []
                side_out_ = [(out - x_enc).detach() + x_enc for out in side_output]
                for (i, out) in enumerate(side_out_):
                    out_decode = self.decode(out)
                    weight = 1 if i != len(side_out_)-1 else 5
                    loss_ = torch.norm(out_decode - x, 2, dim=-1) + distance_loss(out_decode, x)
                    loss_trans.append(weight * loss_)
                loss += torch.stack(loss_trans).mean(dim=0)
        return loss

    def forward(self, x):
        if not self.training:
            return self.forward_test(x)
        else:
            return self.forward_train(x)
        
        