import torch
from dalle2_pytorch import DALLE2, DiffusionPrior, DiffusionPriorTrainer, DecoderTrainer
import matplotlib.pyplot as plt
from dalle2_pytorch.train_configs import DiffusionPriorConfig, TrainDiffusionPriorConfig, TrainDecoderConfig
from torchvision.transforms import ToPILImage
from train_prior import make_model
from decoder import decoder_params1,decoder_params
#from prior import diffusion_prior
from clip import tokenize
#
device = torch.device('cuda')
configPrior = TrainDiffusionPriorConfig.from_json_path("train_prior_config.json")
Prior = make_model(configPrior.prior, configPrior.train)
Prior.load("last_check.pth")
# def make_prior(
#     prior_config: DiffusionPriorConfig, checkpoint_path: str, device: str = None
# ):
#     # create model from config
#     diffusion_prior = prior_config.create()
#     state_dict = torch.load(checkpoint_path, map_location="cpu")
#     diffusion_prior.load_state_dict(state_dict)
#     diffusion_prior.eval()
#     diffusion_prior.to(device)
#
#     if device == "cpu":
#         diffusion_prior.float()
#     return diffusion_prior
# # load entire config
# train_config = TrainDiffusionPriorConfig.from_json_path("train_prior_config.json")
# prior_config = train_config.prior
#
# # load model
# prior = make_prior(prior_config=prior_config, checkpoint_path="best_checkpoint.pth", device="cuda")
# Prior.load("latest_checkpoint.pth")
Decoder = decoder_params1
Decoder.load("Big_decoder.pth")

# configDecoder = TrainDecoderConfig.from_json_path("train_decoder_config.json")
# Decoder = configDecoder.decoder.create().cuda()
# Decoder.load_state_dict(torch.load("best_decoder.pth", map_location='cpu'),strict=False)

#trainer.load("latest.pth")
#print(isinstance(Prior, DiffusionPriorTrainer))

# tokenize the text
#tokenized_text = tokenize("flower")
# predict an embedding
#predicted_embedding = Prior.sample(tokenized_text, cond_scale=1.0)

#print(predicted_embedding[0].shape)
#plt.imshow(predicted_embedding.cpu())
#plt.show()

dalle2 = DALLE2(
    prior = Prior.diffusion_prior,
    decoder = Decoder.decoder,
    prior_num_samples = 2,
)

images = dalle2(
    ['bucket'],
    cond_scale = 1. # classifier free guidance strength (> 1 would strengthen the condition)
).cpu()
print(images[0].shape)
plt.imshow(images[0].permute(1, 2, 0))
plt.show()

for img in images:
    img = ToPILImage()(img)
    img.show()