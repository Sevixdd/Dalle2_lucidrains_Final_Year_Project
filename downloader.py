from img2dataset import download
import shutil
import os

output_tar = os.path.abspath("media/sevi/New Volume/FYP/datasets/dataset")

def main():
  if os.path.exists(output_tar):
      shutil.rmtree(output_tar)
  #mscoco.parquet
  download(
    processes_count=4,
    thread_count=64,
    url_list="mscoco",
    image_size=256,
    output_folder=output_tar ,
    output_format="webdataset",
    input_format="json",
    url_col="image_urls",
    caption_col="captions",
    enable_wandb=False,
    number_sample_per_shard=1000,
    distributor="multiprocessing",
  )

if __name__ == "__main__":
  main()