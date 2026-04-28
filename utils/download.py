import os
import urllib.request

def download_sam_checkpoint():
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    save_dir = "./checkpoints"
    filename = "sam_vit_b_01ec64.pth"
    save_path = os.path.join(save_dir, filename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(save_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, save_path)
        print("Done!")
    else:
        print(f"{filename} already exists.")

if __name__ == "__main__":
    download_sam_checkpoint()