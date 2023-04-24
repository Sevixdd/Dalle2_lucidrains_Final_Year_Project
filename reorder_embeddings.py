
import os

from embedding_dataset_reordering import reorder_embeddings

def test_reorder():
    # TODO: Make this not half-assed. Or don't.
    reorder_embeddings(
        embeddings_folder = "/home/sevi/datasets/sbucaption/embedings/img_emb",
        metadata_folder = "/home/sevi/datasets/sbucaption/index/metadata",
        output_folder ="/home/sevi/datasets/sbucaption/reorder",
        index_width = 3,
        output_shard_width=3,
     )
if __name__ == "__main__":
    test_reorder()