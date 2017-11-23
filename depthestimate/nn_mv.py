import sys
import tensorflow as tf
import numpy as np
import cv2
import tflearn
import random
import math
import os
#os.system("chmod +w /unsullied/sharefs/wangmengdi/wangmengdi")
import time
import zlib
import socket
import threading
import Queue
import sys
import tf_nndistance_1 as tf_nndistance
import cPickle as pickle

#from BatchFetcher import *
#from BatchFetcher_1 import *
#from BatchFetcher2 import *
from BatchFetcherPoke import *

lastbatch=None
lastconsumed=FETCH_BATCH_SIZE
LR_DEFAULT=3e-5

def fetch_batch():
	global lastbatch,lastconsumed
	if lastbatch is None or lastconsumed+BATCH_SIZE>FETCH_BATCH_SIZE:
		lastbatch=fetchworker.fetch()
		lastconsumed=0
	ret=[i[lastconsumed:lastconsumed+BATCH_SIZE] for i in lastbatch]
	lastconsumed+=BATCH_SIZE
	return ret
def stop_fetcher():
	fetchworker.shutdown()

def build_mv_graph(resourceid,lr):
	"""
	Build multi-view graph 
	"""
	with tf.device('/gpu:%d'%resourceid):
		tflearn.init_graph(seed=1029,num_cores=2,gpu_memory_fraction=0.9,soft_placement=True)
		img_inp=tf.placeholder(tf.float32,shape=(BATCH_SIZE,NUM_VIEW,HEIGHT,WIDTH,4),name='img_inp')
		pt_gt=tf.placeholder(tf.float32,shape=(BATCH_SIZE,POINTCLOUDSIZE,3),name='pt_gt')

		# transpose views: (N,V,H,W,C) -> (V,N,H,W,C)
		#img_inp = tf.transpose(img_inp, perm=[1,0,2,3,4])

		view_pool_enc = tf.Variable(tf.zeros([NUM_VIEW, BATCH_SIZE, 3, 4, 512]),name='PoolEnc')
		view_pool_x3 = tf.Variable(tf.zeros([NUM_VIEW, BATCH_SIZE, 24, 32, 128]),name='PoolX3')
		view_pool_x4 = tf.Variable(tf.zeros([NUM_VIEW, BATCH_SIZE, 12, 16, 256]),name='PoolX4')
		view_pool_x5 = tf.Variable(tf.zeros([NUM_VIEW, BATCH_SIZE, 6, 8, 512]),name='PoolX5')
		for viewIdx in range(NUM_VIEW):
			# set reuse True for viewIdx > 0 for weight sharing
			reuse = (viewIdx != 0)
			x=tf.gather(img_inp, indices=viewIdx, axis=1) # each (N,H,W,C)
#192 256
			x=tflearn.layers.conv.conv_2d(x,16,(3,3),scope='Conv2D',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x=tflearn.layers.conv.conv_2d(x,16,(3,3),scope='Conv2D_1',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x0=x
			x=tflearn.layers.conv.conv_2d(x,32,(3,3),scope='Conv2D_2',strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
#96 128
			x=tflearn.layers.conv.conv_2d(x,32,(3,3),scope='Conv2D_3',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x=tflearn.layers.conv.conv_2d(x,32,(3,3),scope='Conv2D_4',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x1=x
			x=tflearn.layers.conv.conv_2d(x,64,(3,3),scope='Conv2D_5',strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
#48 64
			x=tflearn.layers.conv.conv_2d(x,64,(3,3),scope='Conv2D_6',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x=tflearn.layers.conv.conv_2d(x,64,(3,3),scope='Conv2D_7',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x2=x
			x=tflearn.layers.conv.conv_2d(x,128,(3,3),scope='Conv2D_8',strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
#24 32
			x=tflearn.layers.conv.conv_2d(x,128,(3,3),scope='Conv2D_9',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x=tflearn.layers.conv.conv_2d(x,128,(3,3),scope='Conv2D_10',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x3=x
			tf.scatter_update(view_pool_x3, viewIdx, x3)
			#view_pool_x3[viewIdx,:,:,:,:] = x3
			#tf.concat([view_pool_x3, tf.expand_dims(x3, axis=0)], 0)

			x=tflearn.layers.conv.conv_2d(x,256,(3,3),scope='Conv2D_11',strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
#12 16
			x=tflearn.layers.conv.conv_2d(x,256,(3,3),scope='Conv2D_12',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x=tflearn.layers.conv.conv_2d(x,256,(3,3),scope='Conv2D_13',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x4=x
			tf.scatter_update(view_pool_x4, viewIdx, x4)
			#view_pool_x4[viewIdx,:,:,:,:] = x4
			#tf.concat([view_pool_x4, tf.expand_dims(x4, axis=0)], 0)

			x=tflearn.layers.conv.conv_2d(x,512,(3,3),scope='Conv2D_14',strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
#6 8
			x=tflearn.layers.conv.conv_2d(x,512,(3,3),scope='Conv2D_15',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x=tflearn.layers.conv.conv_2d(x,512,(3,3),scope='Conv2D_16',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x=tflearn.layers.conv.conv_2d(x,512,(3,3),scope='Conv2D_17',strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
			x5=x
			tf.scatter_update(view_pool_x5, viewIdx, x5)
			#view_pool_x5[viewIdx,:,:,:,:] = x5
			#tf.concat([view_pool_x5, tf.expand_dims(x5, axis=0)], 0)

			x=tflearn.layers.conv.conv_2d(x,512,(5,5),scope='Conv2D_18',strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse)
#3 4
			# end of encoder, one x vector for one view image
			tf.scatter_update(view_pool_enc, viewIdx, x)
			#print(x.shape)
			#view_pool_enc[viewIdx,:,:,:,:] = x
			#tf.concat([view_pool_enc, tf.expand_dims(x, axis=0)], 0) # view_pool_enc ~ (V,N,H,W,C)

		x = tf.reduce_max(view_pool_enc, axis=0, name='enc_vp')
		#x = view_pool(view_pool_enc, 'enc_vp')
		#print(view_pool_enc.shape)
		#print(x.shape)

		x_additional=tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
		x_additional=tflearn.layers.core.fully_connected(x_additional,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
		x_additional=tflearn.layers.core.fully_connected(x_additional,256*3,activation='linear',weight_decay=1e-3,regularizer='L2')
		x_additional=tf.reshape(x_additional,(BATCH_SIZE,256,3))
		x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')

		x5 = tf.reduce_max(view_pool_x5, axis=0, name='x5_vp')
		#x5 = view_pool(view_pool_x5, 'x5_vp')

		x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x5))
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')

		x4 = tf.reduce_max(view_pool_x4, axis=0, name='x4_vp')
		#x4 = view_pool(view_pool_x4, 'x4_vp')

		x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x4))
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')

		x3 = tf.reduce_max(view_pool_x3, axis=0, name='x3_vp')
		#x3 = view_pool(view_pool_x3, 'x3_vp')

		x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x3))
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,3,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.reshape(x,(BATCH_SIZE,32*24,3))
		x=tf.concat([x_additional,x],axis=1)
		x=tf.reshape(x,(BATCH_SIZE,OUTPUTPOINTS,3))

		#dists_forward,_,dists_backward,_=tf_nndistance.nn_distance(pt_gt,x)
		dists_forward,dists_backward=tf_nndistance.nn_distance(pt_gt,x)
		mindist=dists_forward
		dist0=mindist[0,:]
		dists_forward=tf.reduce_mean(dists_forward)
		dists_backward=tf.reduce_mean(dists_backward)
		loss_nodecay=(dists_forward+dists_backward/2.0)*10000
		loss=loss_nodecay+tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1
		batchno = tf.Variable(0, dtype=tf.int32)
		optimizer = tf.train.AdamOptimizer(lr*BATCH_SIZE/FETCH_BATCH_SIZE).minimize(loss,global_step=batchno)
		batchnoinc=batchno.assign(batchno+1)
	return img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0

def view_pool(view_features, name):
	vp = tf.reduce_max(view_features, axis=0, name=name)

def load_weights(sess, weightsfile):
	loaddict={}
	fin=open(weightsfile,'rb')
	while True:
		try:
			v,p=pickle.load(fin)
		except EOFError:
			break
		loaddict[v]=p
	fin.close()
	for t in tf.trainable_variables():
		print('Loading weights: '+t.name)
		if t.name not in loaddict:
			print 'missing',t.name
		else:
			sess.run(t.assign(loaddict[t.name]))
			del loaddict[t.name]
	for k in loaddict.iteritems():
		if k[0]!='Variable:0':
			print 'unused',k
	print('Weights are loaded sucessfully ^.<')
	return 0

def mvfinetune(resourceid,keyname,weightsfile,batch_number,lr):
	if not os.path.exists(dumpdir):
		os.system("mkdir -p %s"%dumpdir)
	img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_mv_graph(resourceid,lr)
	config=tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.90
	config.gpu_options.allow_growth=True
	config.allow_soft_placement=True
	saver=tf.train.Saver()
	print("Fine tuning with %d batches:"%batch_number)
	with tf.Session(config=config) as sess,\
				open('%s/%s.log'%(dumpdir,keyname),'a') as fout:
		sess.run(tf.global_variables_initializer())
		print("Loading weights from "+weightsfile)
		load_weights(sess, weightsfile)
		sess.run(batchno.assign(0))  # restore this variable to 0. otherwise, it is 300000 in the pretrained model
		trainloss_accs=[0,0,0]
		trainloss_acc0=1e-9
		validloss_accs=[0,0,0]
		validloss_acc0=1e-9
		lastsave=time.time()
		bno=sess.run(batchno)
		fetchworker.bno=bno//(FETCH_BATCH_SIZE/BATCH_SIZE)
		fetchworker.start()
		while bno<batch_number:
			t0=time.time()
			data,ptcloud,validating=fetch_batch()
			t1=time.time()
			validating=validating[0]!=0
			if not validating:
				_,pred,total_loss,trainloss,trainloss1,trainloss2,distmap_0=sess.run([optimizer,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
					feed_dict={img_inp:data,pt_gt:ptcloud})
				trainloss_accs[0]=trainloss_accs[0]*0.99+trainloss
				trainloss_accs[1]=trainloss_accs[1]*0.99+trainloss1
				trainloss_accs[2]=trainloss_accs[2]*0.99+trainloss2
				trainloss_acc0=trainloss_acc0*0.99+1
			else:
				_,pred,total_loss,validloss,validloss1,validloss2,distmap_0=sess.run([batchnoinc,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
					feed_dict={img_inp:data,pt_gt:ptcloud})
				validloss_accs[0]=validloss_accs[0]*0.997+validloss
				validloss_accs[1]=validloss_accs[1]*0.997+validloss1
				validloss_accs[2]=validloss_accs[2]*0.997+validloss2
				validloss_acc0=validloss_acc0*0.997+1
			t2=time.time()
			down=2

			bno=sess.run(batchno)
			if not validating:
				showloss=trainloss
				showloss1=trainloss1
				showloss2=trainloss2
			else:
				showloss=validloss
				showloss1=validloss1
				showloss2=validloss2
			print >>fout,bno,trainloss_accs[0]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,showloss,showloss1,showloss2,validloss_accs[0]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss
			if bno%128==0:
				fout.flush()
			if time.time()-lastsave>900:
				saver.save(sess,'%s/'%dumpdir+keyname+".ckpt")
				lastsave=time.time()
			print bno,'t',trainloss_accs[0]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,'v',validloss_accs[0]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss,t1-t0,t2-t1,time.time()-t0,fetchworker.queue.qsize()
		saver.save(sess,'%s/'%dumpdir+keyname+".ckpt") 

def main(resourceid,keyname,lr):
	if not os.path.exists(dumpdir):
		os.system("mkdir -p %s"%dumpdir)
	img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_mv_graph(resourceid,lr)
	config=tf.ConfigProto()
	config.gpu_options.allow_growth=True
	config.allow_soft_placement=True
	saver=tf.train.Saver()
	with tf.Session(config=config) as sess,\
				open('%s/%s.log'%(dumpdir,keyname),'a') as fout:
		sess.run(tf.global_variables_initializer())
		trainloss_accs=[0,0,0]
		trainloss_acc0=1e-9
		validloss_accs=[0,0,0]
		validloss_acc0=1e-9
		lastsave=time.time()
		bno=sess.run(batchno)
		fetchworker.bno=bno//(FETCH_BATCH_SIZE/BATCH_SIZE)
		fetchworker.start()
		while bno<BATCH_NUMBER:
			t0=time.time()
			data,ptcloud,validating=fetch_batch()
			t1=time.time()
			validating=validating[0]!=0
			if not validating:
				_,pred,total_loss,trainloss,trainloss1,trainloss2,distmap_0=sess.run([optimizer,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
					feed_dict={img_inp:data,pt_gt:ptcloud})
				trainloss_accs[0]=trainloss_accs[0]*0.99+trainloss
				trainloss_accs[1]=trainloss_accs[1]*0.99+trainloss1
				trainloss_accs[2]=trainloss_accs[2]*0.99+trainloss2
				trainloss_acc0=trainloss_acc0*0.99+1
			else:
				_,pred,total_loss,validloss,validloss1,validloss2,distmap_0=sess.run([batchnoinc,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
					feed_dict={img_inp:data,pt_gt:ptcloud})
				validloss_accs[0]=validloss_accs[0]*0.997+validloss
				validloss_accs[1]=validloss_accs[1]*0.997+validloss1
				validloss_accs[2]=validloss_accs[2]*0.997+validloss2
				validloss_acc0=validloss_acc0*0.997+1
			t2=time.time()
			down=2

			bno=sess.run(batchno)
			if not validating:
				showloss=trainloss
				showloss1=trainloss1
				showloss2=trainloss2
			else:
				showloss=validloss
				showloss1=validloss1
				showloss2=validloss2
			print >>fout,bno,trainloss_accs[0]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,showloss,showloss1,showloss2,validloss_accs[0]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss
			if bno%128==0:
				fout.flush()
			if time.time()-lastsave>900:
				saver.save(sess,'%s/'%dumpdir+keyname+".ckpt")
				lastsave=time.time()
			print bno,'t',trainloss_accs[0]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,'v',validloss_accs[0]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss,t1-t0,t2-t1,time.time()-t0,fetchworker.queue.qsize()
		saver.save(sess,'%s/'%dumpdir+keyname+".ckpt") 

def dumppredictions(resourceid,keyname,valnum):
	img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_mv_graph(resourceid,LR_DEFAULT)
	config=tf.ConfigProto()
	config.gpu_options.allow_growth=True
	config.allow_soft_placement=True
	saver=tf.train.Saver()
	fout = open("%s/%s.v.pkl"%(dumpdir,keyname),'wb')
	with tf.Session(config=config) as sess:
		#sess.run(tf.initialize_all_variables())
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,"%s/%s.ckpt"%(dumpdir,keyname))
		fetchworker.bno=0
		fetchworker.start()
		cnt=0
		for i in xrange(0,BATCH_NUMBER):
			t0=time.time()
			data,ptcloud,validating=fetch_batch()
			validating=validating[0]!=0
			if not validating:
				continue
			cnt+=1
			pred,distmap=sess.run([x,mindist],feed_dict={img_inp:data,pt_gt:ptcloud})
			pickle.dump((i,data,ptcloud,pred,distmap),fout,protocol=-1)
			print i,'time',time.time()-t0,cnt
			if cnt>=valnum:
				break
	fout.close()

def testpredictions(resourceid,keyname,valnum,modeldir):
	img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_mv_graph(resourceid,LR_DEFAULT)
	config=tf.ConfigProto()
	config.gpu_options.allow_growth=True
	config.allow_soft_placement=True
	saver=tf.train.Saver()
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,"%s/%s.ckpt"%(modeldir,keyname))
		fetchworker.bno=0
		fetchworker.start()
		cnt=0
		for i in xrange(0,valnum):
			t0=time.time()
			data,ptcloud,validating=fetch_batch()
			pred=sess.run(x,feed_dict={img_inp:data})
			for j in range(0, BATCH_SIZE):
				outfile="%s/pts_"%dumpdir+str(i)+'_'+str(j)+'.txt'
				np.savetxt(outfile,pred[j],fmt='%8.6f',delimiter=' ',newline='\n')
			print i,'time',time.time()-t0,cnt

def exportpkl(resourceid,keyname,modeldir):
	img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_mv_graph(resourceid,LR_DEFAULT)
	config=tf.ConfigProto()
	config.gpu_options.allow_growth=True
	config.allow_soft_placement=True
	saver=tf.train.Saver()
	fout = open("%s/twobranch_%s.pkl"%(dumpdir,keyname),'wb')
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,"%s/%s.ckpt"%(modeldir,keyname))
		for t in tf.trainable_variables():
			print('Saving weights: '+t.name)
			pickle.dump((t.name, sess.run(t)),fout,protocol=-1)

if __name__=='__main__':
	resourceid = 0
	datadir,dumpdir,cmd,valnum,lr="data","dump","predict",3,LR_DEFAULT
	for pt in sys.argv[1:]:
		if pt[:5]=="data=":
			datadir = pt[5:]
		elif pt[:5]=="dump=":
			dumpdir = pt[5:]
		elif pt[:4]=="num=":
			valnum = int(pt[4:])
		elif pt[:4]=="pkl=":
			weightsfile = pt[4:]
		elif pt[:6]=="model=":
			modeldir = pt[6:]
		elif pt[:4]=="bno=":
			batchno = int(pt[4:])
		elif pt[:3]=="lr=":
			lr = np.float32(pt[3:])
		else:
			cmd = pt
	if datadir[-1]=='/':
		datadir = datadir[:-1]
	if dumpdir[-1]=='/':
		dumpdir = dumpdir[:-1]
	#assert os.path.exists(datadir),"data dir not exists"
	os.system("mkdir -p %s"%dumpdir)
	fetchworker=BatchFetcher(datadir)
	print "datadir=%s dumpdir=%s num=%d cmd=%s started"%(datadir,dumpdir,valnum,cmd)
	
	keyname=os.path.basename(__file__).rstrip('.py')
	try:
		if cmd=="train":
			main(resourceid,keyname,lr)
		elif cmd=="predict":
			dumppredictions(resourceid,keyname,valnum)
		elif cmd=="test":
			testpredictions(resourceid,keyname,valnum,modeldir)
		elif cmd=="exportpkl":
			exportpkl(resourceid,keyname,modeldir)
		elif cmd=="mvfinetune":
			mvfinetune(resourceid,keyname,weightsfile,batchno,lr)
		else:
			assert False,"format wrong"
	finally:
		stop_fetcher()
