from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter, \
    OpenClipAdapter, DecoderTrainer, CLIP

# do above for many steps ...

# decoder (with unet)
unet1 = Unet(
    dim = 128,
    image_embed_dim = 768,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8),
    attn_heads=8,
    attn_dim_head=64,
    text_embed_dim = 768,
    cond_on_text_encodings = True  # set to True for any unets that need to be conditioned on text encodings (ex. first unet in cascade)
).cuda()


decoder_params = Decoder(
    unet = unet1,
    image_sizes = [128],
    clip = OpenAIClipAdapter("ViT-L/14"),
    timesteps = 1000,
    sample_timesteps = 64,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.1
).cuda()
decoder_params1 = DecoderTrainer(
    decoder_params,
    lr = 1e-4,
    wd = 1e-2,
    ema_beta = 0.99,
    ema_update_after_step = 1000,
    ema_update_every = 10,

)
