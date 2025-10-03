import os
import numpy as np
import argparse
import os
from collections import defaultdict 

def merge_embs(emb_dir):
    data = defaultdict(list)
    indices = []
    for file_name in os.listdir(emb_dir):
        if file_name.startswith(f'cell_embs') and file_name.endswith('.npz') and 'rank' in file_name and os.access(os.path.join(emb_dir, file_name), os.W_OK):
            npz = np.load(os.path.join(emb_dir, file_name))
            for key in npz.keys():
                data[key].append(npz[key])
            print(f"Removing {os.path.join(emb_dir, file_name)}")
            os.remove(os.path.join(emb_dir, file_name))
        
    if len(data) != 0:
        print(f"Merging embeddings ...")
        for key in data.keys():
            data[key] = np.concatenate(data[key], axis=0)
        
        indices = data['batch_indices']
        for key in data.keys():
            if len(indices) == len(data[key]):
                data[key] = data[key][indices.argsort()]
        
        for key in data.keys():
            np.save(os.path.join(emb_dir, f'{key}.npy'), data[key])
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path", type=str, default=None, help="path to the dataset", required=True)
    args = parser.parse_args()

    # iterate over all the child directories recursively and merge the embedddings:
    for root, dirs, files in os.walk(args.emb_path):
        merge_embs(root)

