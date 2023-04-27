from img2dataset import download
import shutil
import os

output_tar = os.path.abspath("/home/sevi/datasets/cc3m")

def main():
  if os.path.exists(output_tar):
      shutil.rmtree(output_tar)
  #mscoco.parquet
  download(
    processes_count=8,
    thread_count=64,
    url_list="/home/sevi/datasets/Image_Labels_Subset_Train_GCC-Labels-training.tsv",
    image_size=256,
    output_folder=output_tar ,
    output_format="webdataset",
    input_format="tsv",
    url_col="url",
    caption_col="caption",
    enable_wandb=True,
    number_sample_per_shard=1000,
  )

if __name__ == "__main__":
  main()