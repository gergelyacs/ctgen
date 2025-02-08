"""
    Sampling + Compressive Sensing (CS) 
    We exploit that the HR and UC signals are sparse in the frequency domain (and NOT in the time domain).
    
    PROBLEM: The approach works if we take random samples in the time domain, but diffusion cannot generate
    samples at random time points as accurately as at regular time points (along a grid) due to the lack of 
    spatial correlation. On the other hand, the diffusion model can generate samples at regular time points
    but the CS with fourier is extremely slow on regular time points.
"""

import torch
from torch import nn
import numpy as np
from utils import idct1, dct1, dict2class
from pylbfgs import owlqn
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class CS(nn.Module):
    def __init__(self, cfg : dict, input_size : int):
        super().__init__()

        # needed by LDM
        self.latent_size = cfg.latent_size

        # idxs: torch.from_numpy(np.sort(np.random.choice(input_size, cfg.cs.latent_size, replace=False))
        #idxs: torch.arange(0, input_size, input_size // cfg.cs.latent_size),
        _params = {'idxs' :  torch.from_numpy(np.sort(np.random.choice(input_size, cfg.latent_size, replace=False))),
                'input_size' : input_size,
                'scale_factor' : 100, #input_size ** 0.5,
                'latent_size' : self.latent_size,
                'ORTHANTWISE_C' : cfg.ORTHANTWISE_C}

        self.params = dict2class(**_params)
       
        self.b = None        
    
    def evaluate(self, x, g, step):
        """An in-memory evaluation callback for owlqn (limited-memory BFGS).
        """
        # we want to return two things:
        # (1) the norm squared of the residuals, sum((Ax-b).^2), and
        # (2) the gradient 2*A'(Ax-b)

        # extract samples: we solve the linear equation system in the frequency domain
        # the unknown vector x is the DCT of the original signal
        Ax = idct1(x.reshape(self.params.input_size))[self.params.idxs]

        # calculate the residual Ax-b and its 2-norm squared   
        Axb = Ax - self.b
        fx = np.sum(np.power(Axb, 2))

        # project residual vector onto empty vector
        Axb2 = np.zeros(self.params.input_size)
        Axb2[self.params.idxs] = Axb  

        # Compute gradient of (Ax-b)^2: 
        # A'(Ax-b) is just the dct of Axb2
        AtAxb2 = 2 * dct1(Axb2)
      
        # copy over the gradient vector
        np.copyto(g, np.expand_dims(AtAxb2, axis = 1))
       
        return fx

    def progress(self, x, g, fx, xnorm, gnorm, step, k, ls):
        """Just display the current iteration.
        """
        #logger.info('Iteration %d ' % k)
        return 0
       
    def encode(self, x: torch.Tensor):
        return x[:, :, self.params.idxs]
    
    def decode(self, z: torch.Tensor):
        self.b = z
        # reconstruct FHR and UC separately
        x = torch.zeros(z.shape[0], z.shape[1], self.params.input_size)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                self.b = z[i, j, :].cpu().detach().numpy()
                self.b *= self.params.scale_factor
                x_hat = idct1(owlqn(self.params.input_size, self.evaluate, self.progress, self.params.ORTHANTWISE_C).reshape(self.params.input_size))
                x_hat /= self.params.scale_factor
                x[i, j, :] = torch.from_numpy(x_hat)
        return x
    
    def forward(self, x: torch.Tensor):
        return (self.decode(self.encode(x)), ) # due to compatibility with other models
    
    def get_latent(self, x: torch.Tensor):
        return self.encode(x)
    
    def validate(self, testloader, save_path=None, val_hook=None):      
        # start evaluation
        reconstructed_samples = torch.empty(0)
        orig_samples = torch.empty(0)

        total_loss = 0
        for x in tqdm(testloader, desc="Validation"):
            x_hat = self.forward(x)[0]

            # compute loss
            total_loss += torch.abs(x - x_hat).mean()    

            # concat
            reconstructed_samples = torch.cat((reconstructed_samples, x_hat.cpu()), dim=0)
            orig_samples = torch.cat((orig_samples, x.cpu()), dim=0)

        if val_hook is not None:
            val_hook(0, orig_samples, reconstructed_samples, 'cs', idxs=np.random.choice(len(orig_samples), 10, replace=False))
        
        logger.info ("Average L1 error:", total_loss.item() / len(testloader))  
      
    def save(self, save_path):
        # create dictionary from class variables
        torch.save(self.params.__dict__, save_path)

    def load(self, load_path):
        self.params = dict2class(torch.load(load_path))


    
    