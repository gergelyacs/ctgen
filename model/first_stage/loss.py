import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import ResBlock as ResidualBlock
#from model.first_stage import DiagonalGaussianDistribution
from utils import calc_conv_out_size
from utils import sdct_torch, isdct_torch, dct_decompose, merge_freqs
from model.first_stage.focal_freq_loss import FocalFrequencyLoss as FFL

import logging

logger = logging.getLogger(__name__)

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
 
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

## FIXME: Discriminator is not saved in the model checkpoint!!!
class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 2, n_layers: int = 2, input_size: int = 4096,
                 start_channels: int = 64, residual_blocks: bool = False):
        """
        Patch Discriminator for VQ-GAN and VAE, with option to add residual blocks to increase
        model capacity. The output is NOT a single probability value, but 
        the logits produced by the last convolutional layer.
   
        Args:
            in_channels: Input channels (Default: 3).
            n_layers: Number of down-sampling layers. Final resolution will be
                image size divided by 2 ** n_layers.
            start_channels: Number of starting channels, which get multiplied up
                until 2 ** n_layers.
            residual_blocks: If True, adds residual blocks between the down-sampling
                layers.
        """
        super().__init__()

        # To calculate the output size of the convolutional layers 
        kernel_size = 4
        stride = 2
        padding = 0

        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, start_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect'),
            nn.LeakyReLU(0.2, True)
        )
        out_sizes = [calc_conv_out_size(input_size, kernel_size, stride, 1, padding)]

        prev_channels = start_channels
        self.blocks = nn.ModuleList([])
        for n in range(1, n_layers):
            channel_mult = min(2 ** n, 4)
            out_channels = start_channels * channel_mult
            self.blocks.append(
                nn.Sequential(
                     # no need to use bias as BatchNorm2d has affine parameters
                    nn.Conv1d(prev_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.2, True), # good for GANs
                    ResidualBlock(out_channels, out_channels) if residual_blocks else nn.Identity()
                )
            )
            out_sizes.append(calc_conv_out_size(out_sizes[-1], kernel_size, stride, 1, padding))
            prev_channels = out_channels

        channel_mult = min(2 ** n_layers, 4)
        out_channels = start_channels * channel_mult
        self.blocks.append(
                nn.Sequential(
                    # stride=1
                    nn.Conv1d(prev_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.2, True), # good for GANs
                )
            )
        out_sizes.append(calc_conv_out_size(out_sizes[-1], kernel_size, 1, 1, padding))
        prev_channels = out_channels

        self.out_conv = nn.Conv1d(prev_channels, 1, kernel_size=kernel_size, stride=1, padding=padding)
        out_sizes.append(calc_conv_out_size(out_sizes[-1], kernel_size, 1, 1, padding))

        self.apply(weights_init)

        logger.info (f"Patch discriminator output dimension with input size {input_size}: (batch_size, 1, {out_sizes[-1]})")

    def forward(self, x: torch.Tensor):
        x = self.init_conv(x)
   
        for block in self.blocks:
            x = block(x)
   
        x = self.out_conv(x)
        #x = torch.sigmoid(x)
   
        return x

# Loss function for VQ-GAN: 
# - reconstruction loss (L1 or L2)
# - codebook loss (embedding loss + commitment loss)
# - adversarial loss (discriminator)


class VQLossFn(nn.Module):
    def __init__(self,
                 rec_loss_type: str = 'L1',
                 codebook_weight: float = 1.,
                 commitment_weight: float = 0.25,
                 disc_weight: float = 1.,
                 disc_in_channels: int = 2,
                 disc_n_layers: int = 2,
                 last_decoder_layer: nn.Module = None):
        """
        A class for computing and combining the different losses used in VQ-VAE
        and VQ-GAN.

        Args:
            rec_loss_type: Loss-type for reconstruction loss, either L1 or L2.
            codebook_weight: Weight for the codebook loss of the vector quantizer.
            commitment_weight: Beta for the commitment loss of the vector quantizer.z
            disc_weight: Weight for the adversarial loss.
            disc_in_channels: Input channels for the discriminator.
            disc_n_layers: Number of layers in the discriminator.
        """
        super().__init__()

        # reconstruction loss (take L1 as it is produces less blurry
        # results according to https://arxiv.org/abs/1611.07004)
        if rec_loss_type.lower() == 'l1':
            self.rec_loss_fn = nn.L1Loss()
        elif rec_loss_type.lower() == 'l2':
            self.rec_loss_fn = nn.MSELoss()
        elif rec_loss_type.lower() == 'huber':
            self.rec_loss_fn = nn.SmoothL1Loss()
        elif rec_loss_type.lower() == 'focal':
            self.rec_loss_fn = FFL(alpha=1.0)
        else:
            raise ValueError(f"Unknown reconstruction loss type '{rec_loss_type}'!")
      
        # embedding loss (including stop-gradient according to
        # the paper https://arxiv.org/abs/1711.00937)
        self.codebook_weight = codebook_weight
        self.commitment_weight = commitment_weight

        # discriminator loss to avoid blurry images, as they
        # are classified as fake as long as they are blurry
        # (according to https://arxiv.org/abs/1611.07004)
        self.disc_weight = disc_weight

        #self.discriminator = Discriminator(disc_in_channels, n_layers=disc_n_layers,
        #                                   residual_blocks=disc_res_blocks
        #                                   ) if disc_weight > 0 else None
        self.discriminator = Discriminator(disc_in_channels, n_layers=disc_n_layers,
                                           ) if disc_weight > 0 else None
        
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002
                                         ) if disc_weight > 0 else None
        self.last_layer = last_decoder_layer

        if self.discriminator is None:
            logger.warning ("WARNING: Discriminator is not used!")

        logger.info ("==> VQ Loss function with rec_loss_type: %s", rec_loss_type)

    def codebook_loss_fn(self, z_e: torch.Tensor, z_q: torch.Tensor):
        """
        Computes the codebook loss with stop-gradient operator, like
        specified in VQ-VAE paper (https://arxiv.org/abs/1711.00937)
        in equation (3). Computes the embedding loss, which optimizes the embeddings,
        and the commitment loss, which optimizes the encoder.

        Args:
            z_e: Encoded image.
            z_q: Quantized encoded image.
            commitment_weight: Scale factor for commitment loss (default and in
                paper: 0.25).

        Returns:
            codebook_loss: Sum of embedding and (scaled) commitment loss.
        """
        embedding_loss = torch.mean((z_q.detach() - z_e) ** 2)
        commitment_loss = torch.mean((z_q - z_e.detach()) ** 2)

        codebook_loss = embedding_loss + self.commitment_weight * commitment_loss

        return codebook_loss

    def calculate_adaptive_weight(self, rec_loss, generator_loss):

        rec_grads = torch.autograd.grad(rec_loss, self.last_layer.weight,
                                        retain_graph=True)[0]
        generator_grads = torch.autograd.grad(generator_loss, self.last_layer.weight,
                                              retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(generator_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        return d_weight
    
    def stft_loss(self, x1, x2):
        sdct1 = sdct_torch(x1, frame_length=16, frame_step=4)
        sdct2 = sdct_torch(x2, frame_length=16, frame_step=4)
   
        sdct1[:, :, 4:, :] = 0
        sdct2[:, :, 4:, :] = 0

        x1_rec = isdct_torch(sdct1, frame_length=16, frame_step=4)
        x2_rec = isdct_torch(sdct2, frame_length=16, frame_step=4)

        return self.rec_loss_fn(x1_rec, x2_rec)
    
  
    def fft_loss(self, x1, x2, cutoff = .05, lr_weight_fhr=0.25, lr_weight_uc=0.25, hr_weight_fhr=0.40, hr_weight_uc=0.10):
        # compute low frequency content

        x1_low, x1_high = dct_decompose(x1, cutoff=0.05)
        x2_low, x2_high = dct_decompose(x2, cutoff=0.05)

        return self.rec_loss_fn(x1_low, x2_low) + self.rec_loss_fn(x1_high, x2_high)
                   

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor,
                z_e: torch.Tensor, z_q: torch.Tensor,
                disc_training: bool = False):
        """
        Computes the final loss including the following sub-losses:
        - reconstruction loss
        - codebook loss for vector quantizer
        #- perceptual loss using LPIPS
        - adversarial loss with discriminator

        Args:
            x_hat: Reconstructed image.
            x: Original image.
            z_e: Encoded image.
            z_q: Quantized encoded image.
            disc_training: If true, also trains the discriminator.

        Returns:
            loss: The combined loss.
            log: A dictionary containing all sub-losses and the total loss.
        """
        device = x.device
        log = {}
        loss = torch.tensor([0.0]).to(device)

        #decompose x_hat along channel dimension into two equal parts
    
        # loss for generator / autoencoder
        rec_loss = self.rec_loss_fn(x_hat, x)
        loss += rec_loss
        log['rec_loss'] = rec_loss.item()

        # fft loss
        #fft_loss_val = fft_loss(x_hat, x, cutoff=0.2, hr_weight=1000)
        #x1_fft = torch_dct.dct(x_hat, norm='ortho')
        #x2_fft = torch_dct.dct(x, norm='ortho')
        #fft_loss_val = self.rec_loss_fn(x1_fft, x2_fft)
        #loss += fft_loss_val
        #log['fft_loss'] = fft_loss_val.item()
        #stft_loss_val = self.stft_loss(x_hat, x)
        #loss += stft_loss_val
        #log['stft_loss'] = stft_loss_val.item()
        #rec_loss = self.fft_loss(x_hat, x, cutoff=0.05)
        
        f = lambda _x : dct_decompose(_x, cutoff=0.05)
        x1_low, x1_high = torch.vmap(f)(x)
        x2_low, x2_high = torch.vmap(f)(x_hat)
        rec_loss_low = nn.L1Loss()(x1_low, x2_low)
        rec_loss_high = 5 * nn.L1Loss()(x1_high, x2_high)
        #loss += rec_loss_low + rec_loss_high
        
        log['rec_loss_low'] = rec_loss_low.item()
        log['rec_loss_high'] = rec_loss_high.item()

        codebook_loss = self.codebook_weight * self.codebook_loss_fn(z_e, z_q)
        loss += codebook_loss
        log['codebook_loss'] = codebook_loss.item()

        # discriminator loss (discriminator is updated in separate step)
        if self.disc_weight > 0 and disc_training:
            disc_out_fake = self.discriminator(x_hat)
            # !!!! 0 -> fake, 1 -> real? or the reverse?
            #target_fake = torch.zeros(disc_out_fake.shape).to(device)
            #generator_loss = nn.functional.binary_cross_entropy(disc_out_fake, target_fake)

            generator_loss = -torch.mean(disc_out_fake)

            generator_loss *= self.disc_weight

            if self.last_layer is not None:
                d_weight = self.calculate_adaptive_weight(rec_loss, generator_loss)
                generator_loss *= d_weight

            loss += generator_loss
            log['generator_loss'] = generator_loss.item()

        log['loss'] = loss.item()

        return loss, log

    def update_discriminator(self, x_hat: torch.Tensor, x: torch.Tensor):
        """
        Updates the discriminator based on the original images and reconstructions.

        Args:
            x_hat: Reconstructed images.
            x: Original images.

        Returns:
            loss: The loss of the discriminator for fake (x_hat) and real (x) images.
            log: A dictionary containing all sub-losses and the total loss.
        """
        log = {}

        self.discriminator.zero_grad()

        disc_out_real = self.discriminator(x)
        disc_out_fake = self.discriminator(x_hat.detach())

        loss = hinge_d_loss(disc_out_real, disc_out_fake)
        loss.backward()

        """ Train all real batch """
        '''
        # Forward pass real batch through D
        disc_out_real = self.discriminator(x)
        target_real = torch.ones(disc_out_real.shape).to(device)

        # Calculate loss on all-real batch
        disc_loss_real = nn.functional.binary_cross_entropy(disc_out_real, target_real)

        # Calculate gradients for D in backward pass
        disc_loss_real.backward()
        log['disc_loss_real'] = disc_loss_real.item()

        """ Train all fake batch """
        # Classify all fake batch with D
        disc_out_fake = self.discriminator(x_hat.detach())
        target_fake = torch.zeros(disc_out_fake.shape).to(device)

        # Calculate D's loss on the all-fake batch
        disc_loss_fake = nn.functional.binary_cross_entropy(disc_out_fake, target_fake)

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        disc_loss_fake.backward()
        log['disc_loss_fake'] = disc_loss_fake.item()
        loss = disc_loss_fake + disc_loss_real
        '''
        log['disc_loss'] = loss.item()

        """ Update the discriminator """
        # Update D
        self.opt_disc.step()

        return loss, log
    
class VAELossFn(nn.Module):
    def __init__(self,
                 rec_loss_type: str = 'L1',
                 disc_weight: float = 1.,
                 disc_in_channels: int = 2,
                 disc_n_layers: int = 2,
                 logvar_init: float = 0.0,
                 kl_weight: float = 1.0,
                 #disc_residual: bool = False,
                 last_decoder_layer: nn.Module = None):

        super().__init__()

        # reconstruction loss (take L1 as it is produces less blurry
        # results according to https://arxiv.org/abs/1611.07004)
        if rec_loss_type.lower() == 'l1':
            self.rec_loss_fn = nn.L1Loss(reduction='none')
        elif rec_loss_type.lower() == 'l2':
            self.rec_loss_fn = nn.MSELoss(reduction='none')
        elif rec_loss_type.lower() == 'huber':
            self.rec_loss_fn = nn.SmoothL1Loss(reduction='none')
        elif rec_loss_type.lower() == 'focal':
            self.rec_loss_fn = FFL(alpha=1.0)
        else:
            raise ValueError(f"Unknown reconstruction loss type '{rec_loss_type}'!")
      
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.kl_weight = kl_weight

        # discriminator loss to avoid blurry images, as they
        # are classified as fake as long as they are blurry
        # (according to https://arxiv.org/abs/1611.07004)
        self.disc_weight = disc_weight
 
        self.discriminator = Discriminator(disc_in_channels, n_layers=disc_n_layers,
                                           ) if disc_weight > 0 else None
        
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002
                                         ) if disc_weight > 0 else None
        self.last_layer = last_decoder_layer

        if self.discriminator is None:
            logger.warning ("WARNING: Discriminator is not used!")

        logger.info ("==> VAE Loss function with rec_loss_type: %s", rec_loss_type)

    def calculate_adaptive_weight(self, nll_loss, generator_loss):

        rec_grads = torch.autograd.grad(nll_loss, self.last_layer.weight,
                                        retain_graph=True)[0]
        generator_grads = torch.autograd.grad(generator_loss, self.last_layer.weight,
                                              retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(generator_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        return d_weight

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor,
                posterior,
                disc_training: bool = False):

        device = x.device
        log = {}
        loss = torch.tensor([0.0]).to(device)

        # loss for generator / autoencoder
        rec_loss = self.rec_loss_fn(x_hat, x)
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        #print ("NLL loss:", nll_loss)

        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        #print ("NLL loss:", nll_loss)
        loss += nll_loss
        log['nll_loss'] = nll_loss.item()
        
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        log['kl_loss'] = kl_loss.item()

        loss += self.kl_weight * kl_loss

        # discriminator loss (discriminator is updated in separate step)
        if self.disc_weight > 0 and disc_training:
            disc_out_fake = self.discriminator(x_hat)

            generator_loss = -torch.mean(disc_out_fake)

            generator_loss *= self.disc_weight

            if self.last_layer is not None:
                d_weight = self.calculate_adaptive_weight(nll_loss, generator_loss)
                generator_loss *= d_weight

            loss += generator_loss
            log['generator_loss'] = generator_loss.item()

        log['total_loss'] = loss.item()

        f = lambda _x : dct_decompose(_x, cutoff=0.05)
        x1_low, x1_high = torch.vmap(f)(x)
        x2_low, x2_high = torch.vmap(f)(x_hat)
        rec_loss_low = nn.L1Loss()(x1_low, x2_low)
        rec_loss_high = 5 * nn.L1Loss()(x1_high, x2_high) 

        log['rec_loss_low'] = rec_loss_low.item()
        log['rec_loss_high'] = rec_loss_high.item()

        return loss, log


    def update_discriminator(self, x_hat: torch.Tensor, x: torch.Tensor):
        """
        Updates the discriminator based on the original images and reconstructions.

        Args:
            x_hat: Reconstructed images.
            x: Original images.

        Returns:
            loss: The loss of the discriminator for fake (x_hat) and real (x) images.
            log: A dictionary containing all sub-losses and the total loss.
        """
        log = {}

        self.discriminator.zero_grad()

        disc_out_real = self.discriminator(x)
        disc_out_fake = self.discriminator(x_hat.detach())

        loss = hinge_d_loss(disc_out_real, disc_out_fake)
        loss.backward()

        log['disc_loss'] = loss.item()

        """ Update the discriminator """
        # Update Discriminator
        self.opt_disc.step()

        return loss, log



