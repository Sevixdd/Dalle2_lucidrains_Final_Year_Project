from dalle2_pytorch.dataloaders import ImageEmbeddingDataset, create_image_embedding_dataloader
import torch
import cv2
from dalle2_pytorch import DALLE2, Unet, Decoder, OpenAIClipAdapter, \
    OpenClipAdapter, DecoderTrainer, CLIP
from clip import tokenize
from torchvision.transforms import transforms
from tqdm import tqdm
from decoder import decoder_params as decoder

epochs = 30
batch_size = 2
num_workers = 1

transform1 = transforms.Compose([
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.75, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
]
)


def preproc(image):
    image = transform1(image)

    return image


dataloader = create_image_embedding_dataloader(
    tar_url="/home/sevi/datasets/sbucaption/subcaptions/{00000..00999}.tar",
    # Uses bracket expanding notation. This specifies to read all tars from 0000.tar to 9999.tar
    img_embeddings_url="/home/sevi/datasets/sbucaption/reorder",
    text_embeddings_url="/home/sevi/datasets/sbucaption/embedings/text_emb",
    # Included if .npy files are not in webdataset. Left out or set to None otherwise
    num_workers=num_workers,
    batch_size=batch_size,
    index_width=3,
    # If a file in the webdataset shard 3 is named 0003039.jpg, we know the shard width is 4 and the last three digits are the index
    extra_keys=["txt"],
    shuffle_num=200,  # Does a shuffle of the data with a buffer size of 200
    shuffle_shards=True,  # Shuffle the order the shards are read in
    resample_shards=False,  # Sample shards with replacement. If true, an epoch will be infinite unless stopped manually
    img_preproc=preproc,

)



clip = OpenAIClipAdapter("ViT-L/14")

trainer = DecoderTrainer(
    decoder=decoder,
    lr=1e-4,
    wd=1e-2,
    ema_beta=0.99,
    use_ema=True,
    ema_update_after_step=1000,
    ema_update_every=10,
    dataloaders=dataloader,

)

device = torch.device('cuda')


def send_to_device(arr):
    return [x.to(device=device, dtype=torch.float) for x in arr]


for epoch in range(epochs):
    print("Training %d epoch" % (epoch))
    i=0
    for i, (img, emb, txt) in enumerate(tqdm(dataloader)):
        img_emb = emb.get('img')
        has_img_embedding = img_emb is not None
        if has_img_embedding:
            img_emb, = send_to_device((img_emb,))
        text_emb = emb.get('text')
        has_text_embedding = text_emb is not None
        if has_text_embedding:
            text_emb, = send_to_device((text_emb,))
        img, = send_to_device((img,))
        print("Training %d batch" % (i))
        # img_emb, text_emb = emb['img'], emb['text']
        forward_params = {}
        if has_img_embedding:
            forward_params['image_embed'] = img_emb
        else:
            # Forward pass automatically generates embedding
            assert clip is not None
            img_embed, img_encoding = clip.embed_image(img)
            forward_params['image_embed'] = img_embed

        #forward_params['text_encodings'] = text_emb
            # Then we need to pass the text instead
        assert clip is not None
        tokenized_texts = tokenize(txt, truncate=True).to(device=device)
        assert tokenized_texts.shape[0] == len(
            img), f"The number of texts ({tokenized_texts.shape[0]}) should be the same as the number of images ({len(img)})"
        text_embed, text_encodings = clip.embed_text(tokenized_texts)
        forward_params['text_encodings'] = text_encodings
        loss = trainer.forward(img, **forward_params, unet_number=1, _device=device)
        print("Loss=%f" % (loss))
        trainer.update()  # update the specific unet as well as its exponential moving average
        if i % 1000 == 0:
            print("save")
            trainer.save("Big_best_decoder.pth")
    print("save")
    trainer.save("Big_decoder.pth")
# do above for many steps


# images = dalle2(
#     ['a butterfly trying to escape a tornado'],
#     cond_scale = 2. # classifier free guidance strength (> 1 would strengthen the condition)
# )
