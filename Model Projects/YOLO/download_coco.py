import os
import zipfile
import urllib.request


def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")


def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


base_dir = "coco2017"
os.makedirs(base_dir, exist_ok=True)

files = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

for name, url in files.items():
    zip_path = os.path.join(base_dir, f"{name}.zip")
    download_file(url, zip_path)
    unzip_file(zip_path, base_dir)