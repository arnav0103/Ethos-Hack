import os   
import shutil 

try:
    shutil.rmtree('./inputs')
    shutil.rmtree('./results')
except:
    pass
os.system("python frame.py")
os.system("python inference_gfpgan.py")