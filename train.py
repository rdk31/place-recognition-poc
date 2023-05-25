import timm
from PIL import Image
from tqdm.auto import tqdm
import faiss
from pathlib import Path
import pickle

from utils import get_data


train_data, test_data = get_data("Eynsham")

model = timm.create_model(
    "convnext_tiny.fb_in22k_ft_in1k",
    pretrained=True,
    num_classes=0,
)
model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

embeddings = faiss.IndexFlatIP(model.num_features)
positions = []

root_dir = Path("Eynsham")
original_images_dir = root_dir / "Images"

for pos, images in tqdm(train_data):
    img = Image.open(original_images_dir / images[0])
    rgb_image = Image.new("RGB", img.size)
    rgb_image.paste(img)

    embedding = model(transforms(rgb_image).unsqueeze(0)).detach().numpy()
    embeddings.add(embedding)

    positions.append(pos)


faiss.write_index(embeddings, "embeddings.index")
with open("positions.pkl", "wb") as f:
    pickle.dump(positions, f)
