import timm
from PIL import Image
from tqdm.auto import tqdm
import faiss
import pandas as pd
from pathlib import Path
import pickle

from utils import get_data, euclidean_distance


train_data, test_data = get_data("Eynsham")

model = timm.create_model(
    "convnext_tiny.fb_in22k_ft_in1k",
    pretrained=True,
    num_classes=0,
)
model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

root_dir = Path("Eynsham")
original_images_dir = root_dir / "Images"

embeddings = faiss.read_index("embeddings.index")
with open("positions.pkl", "rb") as f:
    positions = pickle.load(f)

results = []

for pos, images in tqdm(test_data):
    img = Image.open(original_images_dir / images[0])
    rgb_image = Image.new("RGB", img.size)
    rgb_image.paste(img)

    embedding = model(transforms(rgb_image).unsqueeze(0)).detach().numpy()
    d, i = embeddings.search(embedding, 1)

    found_pos = positions[i[0][0]]
    distance = euclidean_distance(found_pos, pos)
    results.append(distance)

df = pd.DataFrame(results, columns=["distance"])
print(df.describe())
