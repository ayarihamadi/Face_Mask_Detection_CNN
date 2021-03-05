import os.path
from PIL import Image
from tqdm import tqdm

dataset_path = './drive/MyDrive/Extract/faces'

for f in tqdm(os.listdir(dataset_path)):
    f_img = dataset_path + '/' + f
    img = Image.open(f_img)
    img = img.resize((224, 224))
    img.save(f_img)
