from clip_retrieval import clip_inference
import shutil
import os

tar_path = os.path.abspath("/home/sevi/datasets/sbucaption/subcaptions/{00000..00999}.tar")
output_fold_embedings = os.path.abspath("/home/sevi/datasets/sbucaption/embedings")

def clip():
    if os.path.exists(output_fold_embedings ):
        shutil.rmtree(output_fold_embedings )
    clip_inference(

        clip_model="ViT-L/14",
        input_dataset=tar_path,
        output_folder=output_fold_embedings ,
        output_partition_count=1,
        input_format="webdataset",
        enable_metadata=False,
    )

if __name__ == "__main__":
    clip()
