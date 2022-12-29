from img2dataset import download
import shutil
import os

output_dir = os.path.abspath("example_folder")

def main():
  if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
  #mscoco.parquet
  download(
    processes_count=4,
    thread_count=64,
    url_list="mylist.txt",
    image_size=256,
    output_folder=output_dir,
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