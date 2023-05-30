# place-recognition-poc

1. Download [dataset](https://zenodo.org/record/1243106)
2. Unpack it to root dir
3. Install conda env: `conda env create -f environment.yml`, probably have to apply: https://github.com/facebookresearch/faiss/issues/2852
4. Activate conda env: `conda activate place-recognition`
3. Build embeddings: `python train.py`
4. Test embeddings: `python test.py`
