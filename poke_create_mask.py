# Create image_mask of pokemon
# Usage: python poke_create_mask.py data/001 <- pokemon number
# Output files: 001_{}_RGB.png and 001_{}_mask.png
import numpy as np
import os
import scipy.misc
import sys
from PIL import Image

BATCH_SIZE2=30
HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=4096
OUTPUTPOINTS=1024
REEBSIZE=1024

def fetch_image(path2png):
	image = np.array(Image.open(path2png))
	data = np.zeros(image.shape)
	data[:,:,:3] = image[:,:,:3]/255.0
	data[:,:,3] = image[:,:,3]==0
	return data

def loadFile(path, pokenum):
	data=np.zeros((BATCH_SIZE2,HEIGHT,WIDTH,4),dtype='float32')
	for i in range(BATCH_SIZE2):
		path2png = os.path.join(path, pokenum+'_{}.png'.format(i))
		single_data=fetch_image(path2png)
		data[i,:,:,:] = single_data

	depth = np.uint16((data[:,:,:,3]!=0)*255)
	for i in range(BATCH_SIZE2):
		path2 = './'+pokenum+'_{}_RGB.png'.format(i)
		path3 = './'+pokenum+'_{}_mask.png'.format(i)
		scipy.misc.imsave(path2,data[i,:,:,:3]) 
		scipy.misc.imsave(path3,depth[i,:,:])
	return depth

if len(sys.argv)!=2:
	print 'Format: python poke_create_mask.py data/001'
	sys.exit(0)
ifname=sys.argv[1]
pokenum = ifname.split('/')[-1]
loadFile(ifname, pokenum)
