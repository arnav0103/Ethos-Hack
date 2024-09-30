import os   
import shutil 
import requests

def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)

if not os.path.exists('./GFPGANv1.3.pth'):
    download_file('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', './GFPGANv1.3.pth')

try:
    shutil.rmtree('./inputs')
    shutil.rmtree('./results')
except:
    pass
os.system("python frame.py")
os.system("python inference_gfpgan.py")