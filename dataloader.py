# from embedding_reader import EmbeddingReader
#
# embedding_reader = EmbeddingReader(
#     embeddings_folder="/media/sevi/New Volume/FYP/datasets/example/example output",
#     metadata_folder="/media/sevi/New Volume/FYP/datasets/example/example output/index",
#     meta_columns=['image_path', 'caption'],
#     file_format="parquet_npy"
# )
# print("embedding count", embedding_reader.count)
# print("dimension", embedding_reader.dimension)
# print("total size", embedding_reader.total_size)
# print("byte per item", embedding_reader.byte_per_item)
#
# for emb, meta in embedding_reader(batch_size=10 ** 6, start=0, end=embedding_reader.count):
#     print(emb.shape)
#     print(meta["image_path"], meta["caption"])

from dalle2_pytorch.dataloaders import ImageEmbeddingDataset, create_image_embedding_dataloader
from torchvision.transforms import transforms


transform1 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

def preproc(img):
    img= transform1(img)
    return img

#Create a dataloader directly.
dataloader = create_image_embedding_dataloader(
    tar_url="/home/sevi/datasets/example/examples/{00000..00037}.tar", # Uses bracket expanding notation. This specifies to read all tars from 0000.tar to 9999.tar
    img_embeddings_url="/home/sevi/datasets/example/reorder",
    text_embeddings_url="/home/sevi/datasets/example/example output/text_emb",
    # Included if .npy files are not in webdataset. Left out or set to None otherwise
    num_workers=4,
    batch_size=32,
    index_width=4,                                         # If a file in the webdataset shard 3 is named 0003039.jpg, we know the shard width is 4 and the last three digits are the index
    shuffle_num=200,                                       # Does a shuffle of the data with a buffer size of 200
    shuffle_shards=True,                                   # Shuffle the order the shards are read in
    resample_shards=False,
    img_preproc=preproc
)

for img, emb in dataloader:
    print(img.shape)  # torch.Size([32, 3, 256, 256])
    print(emb["img"])  # torch.Size([32, 512])
    # Train decoder only as shown above
