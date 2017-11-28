import sys
import numpy as np
import cv2
import random
import math
import os
import time
import zlib
import socket
import threading
import Queue
import sys
import cPickle as pickle
import show3d
from PIL import Image

NUM_VIEW = 3
TOTAL_NUM_VIEW = 30
FETCH_BATCH_SIZE=30
BATCH_SIZE=30
HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=2600
OUTPUTPOINTS=1024
REEBSIZE=1024
BATCH_NUMBER=120 # Original 300000

class BatchFetcher(threading.Thread):
	def __init__(self, dataname):
		super(BatchFetcher,self).__init__()
		self.queue=Queue.Queue(64)
		self.stopped=False
		self.datadir = dataname
		self.bno=0

	def fetch_single(self, path2png, path2txt):
		image = np.array(Image.open(path2png))
		data = np.zeros(image.shape)
		mask = np.zeros(data[:,:,:3].shape)
		for i in range(3):
			mask[:,:,i] = (image[:,:,3]!=0)
		
		data[:,:,:3] = (image[:,:,:3]*mask + 191*(1-mask))/255.0
		data[:,:,3] = image[:,:,3]==0
		temp1 = path2png.partition('_')[2]
		theta = (int(temp1.partition('.')[0])+11)*12*math.pi/180
		ptcloud = np.loadtxt(path2txt)
		ptcloud = ptcloud.dot(np.array([[np.cos(theta),0, np.sin(theta)],[0,1,0],[-np.sin(theta),0, np.cos(theta)]])).dot(np.array([[np.cos(math.pi/2), np.sin(math.pi/2),0],[-np.sin(math.pi/2), np.cos(math.pi/2),0],[0,0,1]]))		
		repnum = POINTCLOUDSIZE//ptcloud.shape[0] + 1
		ptcloud = np.matlib.repmat(ptcloud,repnum,1)
		ptcloud = ptcloud[0:POINTCLOUDSIZE,:]
		return data, ptcloud
		
	def work(self,bno):
		datalist = os.listdir(self.datadir)
		datalist = [x for x in datalist if x!='0']
		datalist = sorted(datalist)
		data=np.zeros((FETCH_BATCH_SIZE,NUM_VIEW,HEIGHT,WIDTH,4),dtype='float32')
		ptcloud=np.zeros((FETCH_BATCH_SIZE,NUM_VIEW,POINTCLOUDSIZE,3),dtype='float32')		
		validating = np.random.randint(16,size=FETCH_BATCH_SIZE)==0
		for i in range(FETCH_BATCH_SIZE):
			pokenum = datalist[bno]
			firstnum = np.random.randint(TOTAL_NUM_VIEW)
			viewnum = [(firstnum+k*(TOTAL_NUM_VIEW/NUM_VIEW))%(TOTAL_NUM_VIEW) for k in range(NUM_VIEW)]
			#viewnum = random.sample(range(TOTAL_NUM_VIEW), NUM_VIEW)
			for j in range(len(viewnum)):	
				path2png = os.path.join(self.datadir, pokenum, pokenum+'_{}.png'.format(viewnum[j]))
				path2txt = os.path.join(self.datadir, pokenum, pokenum+'.txt')
				single_data, single_ptcloud=self.fetch_single(path2png, path2txt)
				data[i,j,:,:,:] = single_data
				ptcloud[i,j,:,:] = single_ptcloud
 
		return (data,ptcloud,validating)


	def run(self):
		while self.bno<BATCH_NUMBER and not self.stopped:
			self.queue.put(self.work(self.bno%BATCH_NUMBER))
			self.bno+=1
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	def shutdown(self):
		self.stopped=True
		while not self.queue.empty():
			self.queue.get()

if __name__=='__main__':
	dataname = "YTTRBtraindump_220k" #default weird name..
	fetchworker = BatchFetcher(dataname)
	fetchworker.bno=0
	fetchworker.start()
	for cnt in xrange(100):
		data,ptcloud,validating = fetchworker.fetch()
		validating = validating[0]!=0
		assert len(data)==FETCH_BATCH_SIZE
		for i in range(len(data)):
			cv2.imshow('data',data[i])
			while True:
				cmd=show3d.showpoints(ptcloud[i])
				if cmd==ord(' '):
					break
				elif cmd==ord('q'):
					break
			if cmd==ord('q'):
				break


