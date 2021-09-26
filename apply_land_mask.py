import rasterio as rio
import os
import matplotlib.pyplot as plt

inferece_dir = '/media/salman/Windows/Users/salman/PycharmProjects/IceSea/Inference_Journal/'
LM_dir = '/mnt/'

for f in os.listdir(inferece_dir):
    # print(f)
    if f.startswith('S1'):
        id  = f.split('_')[4]
        for l in os.listdir(LM_dir):
            if id in l:

                o = rio.open(inferece_dir + f)
                i = rio.open(LM_dir + l)
                image = o.read()
                LM = i.read()
                print('yes')


