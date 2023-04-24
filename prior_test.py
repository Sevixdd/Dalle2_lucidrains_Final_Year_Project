from dalle2_pytorch.dataloaders import ImageEmbeddingDataset, create_image_embedding_dataloader, make_splits
import torch
import cv2
from dalle2_pytorch import DALLE2, Unet, DiffusionPrior, OpenAIClipAdapter, \
    OpenClipAdapter, DiffusionPriorTrainer, CLIP
from clip import tokenize
from dalle2_pytorch.train_configs import DiffusionPriorConfig, DiffusionPriorTrainConfig
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dalle2_pytorch.dataloaders import get_reader
from prior import diffusion_prior

train_loader = DataLoader
epochs = 3
batch_size = 4
num_workers = 1


clip = OpenAIClipAdapter("ViT-L/14")

device = torch.device('cuda')


def make_model(
    prior_config: DiffusionPriorConfig,
    train_config: DiffusionPriorTrainConfig,
    device: str = None,
):
    # create model from config
    diffusion_prior = prior_config.create()

    # instantiate the trainer
    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=train_config.lr,
        wd=train_config.wd,
        max_grad_norm=train_config.max_grad_norm,
        amp=train_config.amp,
        use_ema=train_config.use_ema,
        device=device,
        warmup_steps=train_config.warmup_steps,
    )

    return trainer

reader = get_reader(text_conditioned=True,
                    img_url="/home/sevi/datasets/reorder",
                    meta_url="/home/sevi/datasets/index/metadata" ,
                    txt_url="/home/sevi/datasets/embedings/text_emb")
make_splits()

for epoch in range(epochs):
    img = img_reader
    txt = text_reader
    loss = trainer(text=txt, image_embed= img)
    trainer.update()
    trainer.save("prior_check.pth")