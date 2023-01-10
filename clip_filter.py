from clip_retrieval import clip_back

def filter():
    clip_back(
        #query="cat",
        #output_folder="/media/sevi/New Volume/FYP/datasets/sbucaption/catfolder",
        #indice_folder="/media/sevi/New Volume/FYP/datasets/sbucaption/index",
        port=1234,
        indices_paths="/home/sevi/Downloads/indices_paths.json",
    )

if __name__ == "__main__":
    filter()

