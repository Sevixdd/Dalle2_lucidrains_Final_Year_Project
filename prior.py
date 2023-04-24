from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter, \
    OpenClipAdapter, DecoderTrainer, CLIP, DiffusionPriorTrainer
from torchvision.transforms import transforms
from tqdm import tqdm


prior_network = DiffusionPriorNetwork(
    dim = 768,
    depth = 12,
    dim_head = 128,
    heads = 12
).cuda()

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = OpenAIClipAdapter("ViT-L/14"),
    timesteps = 1000,
    cond_drop_prob = 0.2
).cuda()

Prior = DiffusionPriorTrainer(
    diffusion_prior,
    lr = 3e-4,
    wd = 1e-2,
    ema_beta = 0.99,
    ema_update_after_step = 1000,
    ema_update_every = 10,
)