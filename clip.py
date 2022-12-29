from clip_retrieval import clip_inference
import shutil
import os

tar_path = os.path.abspath("F:/datasets/00000.tar")
output_fold = os.path.abspath("F:/FYP/embeddings_folder")


def clip():
    if os.path.exists(output_fold):
        shutil.rmtree(output_fold)
    clip_inference(
        clip_model="ViT-L/14",
        input_dataset=tar_path,
        output_folder=output_fold,
        input_format="webdataset",
        enable_wandb=False,
    )


if __name__ == "__main__":
    clip()
