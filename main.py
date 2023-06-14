import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch.nn.functional as F
import torch
from torch.optim import Adam

from model import Unet
from utils import fix_experiment_seed, get_dataloaders, show_image, save_image


def extract(a, t, x_shape):
    """
    Takes a data tensor a and an index tensor t, and returns a new tensor
    whose i^th element is just a[t[i]]. Note that this will be useful when
    we would want to choose the alphas or betas corresponding to different
    indices t's in a batched manner without for loops.
    
    Parameters
    ----------
    a: Tensor, generally of shape (batch_size,)
    t: Tensor, generally of shape (batch_size,)
    x_shape: Shape of the data, generally (batch_size, 3, 32, 32)
    
    Returns:
    out: Tensor of shape (batch_size, 1, 1, 1) generally, the number of 1s are
        determined by the number of dimensions in x_shape.
        out[i] contains a[t[i]]
    """  
    batch_size = t.shape[0]
    out = a.cpu().gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(beta_start, beta_end, timesteps):
  return torch.linspace(beta_start, beta_end, timesteps)

# Hyperparameters taken from Ho et. al for noise scheduling
T = 1000            # Diffusion Timesteps
beta_start = 0.0001 # Starting variance
beta_end = 0.02     # Ending variance

device = "cuda" if torch.cuda.is_available() else "cpu"

betas =  linear_beta_schedule(beta_start, beta_end, T).to(device)
alphas = 1.0 - betas
sqrt_recip_alphas =  torch.sqrt(1.0 / alphas).to(device)
alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0).to(device)
posterior_variance =  ((1. - alphas_cumprod_prev) / (1. - alphas_cumprod)) * betas.to(device)

def q_sample(x_start, t, noise=None):
    """
    Forward Diffusion Sampling Process
    
    Parameters
    ----------
    x_start: Tensor of original images of size (batch_size, 3, 32, 32)
    t: Tensor of timesteps, of shape (batch_size,)
    noise: Optional tensor of same shape as x_start, signifying that the noise to add is already provided.
    
    Returns
    -------
    x_noisy: Tensor of noisy images of size (batch_size, 3, 32, 32)
                x_noisy[i] is sampled from q(x_{t[i]} | x_start[i])
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    x_noisy = sqrt_alphas_cumprod_t.to(device) * x_start.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    return x_noisy

def visualize_diffusion():
  train_dataloader, _ = get_dataloaders(data_root=data_root, batch_size=train_batch_size)
  imgs,_ = next(iter(train_dataloader))
  sample = imgs[3].unsqueeze(0).to(device)
  noisy_images = [sample] + [q_sample(sample, torch.tensor([100 * t + 99]).to(device)) for t in range(10)]
  noisy_images = (torch.cat(noisy_images, dim=0) + 1.) * 0.5
  show_image(noisy_images.clamp(0., 1.), nrow=11)


def p_sample(model, x, t, t_index):
    """
    Given the denoising model, batched input x, and time-step t, returns a slightly denoised sample at time-step t-1
    
    Parameters
    ----------
    model: The denoising (parameterized noise) model
    x: Batched noisy input at time t; size (batch_size, 3, 32, 32)
    t: Batched time steps; size (batch_size,)
    t_index: Single time-step, whose batched version is present in t

    Returns
    -------
    sample: A sample from the distribution p_\theta(x_{t-1} | x_t); mode if t=0
    """
    with torch.no_grad():
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape) 
        
        p_mean = ((x - (betas_t * model(x, t)) / sqrt_one_minus_alphas_cumprod_t )*sqrt_recip_alphas_t).to(device)

        if t_index == 0:
            sample = p_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise = torch.randn_like(x).to(device)
            sample = (p_mean + torch.sqrt(posterior_variance_t) * noise).to(device)
        return sample

def p_sample_loop(model, shape, timesteps):
    """
    Given the model, and the shape of the image, returns a sample from the data distribution by running through the backward diffusion process.
    
    Parameters
    ----------
    model: The denoising model
    shape: Shape of the samples; set as (batch_size, 3, 32, 32)

    Returns
    -------
    imgs: Samples obtained, as well as intermediate denoising steps, of shape (T, batch_size, 3, 32, 32)
    """  
    with torch.no_grad():
        b = shape[0]
        # Start from Pure Noise (x_T)
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(0, timesteps)), desc='Sampling', total=T, leave=False):
            img = p_sample(model, img, torch.tensor(i).repeat(b).to(device), i)
            imgs.append(img.cpu())

        return torch.stack(imgs)

def sample(model, image_size, batch_size=16, channels=3):
    """  
    Returns a sample by running the sampling loop
    """  
    with torch.no_grad():
        return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), timesteps=T)

def p_losses(denoise_model, x_start, t):

    """
    Returns the loss for training of the denoise model

    Parameters
    ----------
    denoise_model: The parameterized model
    x_start: The original images; size (batch_size, 3, 32, 32)
    t: Timesteps (can be different at different indices); size (batch_size,)

    Returns
    -------
    loss: Loss for training the model
    """  
    noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t) 
    loss = F.smooth_l1_loss(predicted_noise, noise)

    return loss

def t_sample(timesteps, batch_size):

    """
    Returns randomly sampled timesteps

    Parameters
    ----------
    timesteps: The max number of timesteps; T
    batch_size: batch_size used in training

    Returns
    -------
    ts: Tensor of size (batch_size,) containing timesteps randomly sampled from 0 to timesteps-1 
    """  
    ts = torch.randint(low=0, high=timesteps, size=(batch_size,)) 
    return ts


if __name__ == '__main__':

    fix_experiment_seed()

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok = True)

    # Training Hyperparameters
    train_batch_size = 64   # Batch Size
    lr = 1e-4         # Learning Rate


    # Define Dataset Statistics
    image_size = 32
    input_channels = 3
    data_root = './data'

    visualize_diffusion()

    model = Unet(
    dim=image_size,
    channels=input_channels,
    dim_mults=(1, 2, 4, 8)
        )

    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    
    epochs = 101

    train_dataloader, test_loader = get_dataloaders(data_root, batch_size=train_batch_size)
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0

        with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")

                optimizer.zero_grad()
                imgs,_ = batch
                batch_size = imgs.shape[0]
                x = imgs.to(device)

                t = t_sample(T, batch_size).to(device) # Randomly sample timesteps uniformly from [0, T-1]

                loss = p_losses(model, x, t)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())
        epoch_loss /= len(train_dataloader)
        epoch_elapsed_time = time.time() - epoch_start_time
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}, loss={epoch_loss:.4f}, time={epoch_elapsed_time:.2f}s")
        if epoch % 10 == 0:
            # Sample and Save Generated Images
            save_image((x + 1.) * 0.5, f"./results/orig_{epoch}.png")
            samples = sample(model, image_size=image_size, batch_size=64, channels=input_channels)
            samples = (torch.Tensor(samples[-1]) + 1.) * 0.5
            save_image(samples, f'./results/samples_{epoch}.png')
    
    show_image(samples)
    weight_path = "model_weights.pt"
    torch.save(model.state_dict(), weight_path)

    model.eval()
    noise_level = 0.1
    t_star = 5  # Modify as required
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i == 0: # take the first batch
                x_test, _ = batch
                x_test = x_test.to(device)

                # add noise to the test samples at timestep t_star
                noise = noise_level * torch.randn_like(x_test)
                x_noisy = q_sample(x_test, t_star, noise)

                # denoise the noisy samples
                predicted_noise = model(x_noisy, t_star)
                x_denoised = x_noisy - predicted_noise

                # plot original, noisy and denoised samples
                plt.figure(figsize=(15, 5))
                for j in range(5): # plot 5 samples from the batch
                    plt.subplot(3, 5, j+1)
                    plt.imshow(x_test[j].permute(1,2,0).cpu())
                    plt.title('Original')
                    plt.axis('off')

                    plt.subplot(3, 5, j+6)
                    plt.imshow(x_noisy[j].permute(1,2,0).cpu())
                    plt.title(f'Noisy, t={t_star}')
                    plt.axis('off')

                    plt.subplot(3, 5, j+11)
                    plt.imshow(x_denoised[j].permute(1,2,0).cpu())
                    plt.title('Denoised')
                    plt.axis('off')
                    plt.show()
                break



