from clip_retrieval import clip_index
from clip_inference import output_fold_embedings

#indexes_sbu = "",
def index():
    clip_index(
        embeddings_folder=output_fold_embedings,
        index_folder="/home/sevi/datasets/sbucaption/index",
        max_index_memory_usage="8G",
    )

if __name__ == "__main__":
    index()

