


def download_medvit_checkpoint(model_size='small', force_download=False):
    """Download official MedViT checkpoint"""
    import urllib.request
    import os
    
    urls = {
        'tiny': 'https://drive.google.com/file/d/14wcH5cm8P63cMZAUHA1lhhJgMVOw_5VQ/view?usp=sharing',
        'small': 'https://drive.google.com/file/d/1Lrfzjf3CK7YOztKa8D6lTUZjYJIiT7_s/view?usp=sharing', 
        'base': 'https://drive.google.com/file/d/1sU-nLpYuCI65h7MjFJKG0yphNAlUFSKG/view?usp=sharing'
    }
    
    if model_size not in urls:
        print(f"❌ Unknown model size: {model_size}")
        return False
    
    filename = f"medvit_{model_size}_official.pth"
    url = urls[model_size]
    
    if os.path.exists(filename) and not force_download:
        print(f"✅ {filename} already exists")
        return filename
    
    try:
        print(f"📥 Downloading {filename} from official source...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded {filename}")
        return filename
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False