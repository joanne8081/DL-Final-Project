import tensorflow as tf
from tensorflow.python.framework import ops
from BatchFetchPoke2 import *
#nn_distance_module=tf.load_op_library('./tf_nndistance_so.so')

def nn_distance(xyz1,xyz2):
	'''
Computes the distance of nearest neighbors for a pair of point clouds
input: xyz1: (batch_size,#points_1,3)  the first point cloud
input: xyz2: (batch_size,#points_2,3)  the second point cloud
output: dist1: (batch_size,#point_1)   distance from first to second
output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
output: dist2: (batch_size,#point_2)   distance from second to first
output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
	'''
	fshape1 = tf.shape(xyz1)  # (B, M1, N=3)
	fshape2 = tf.shape(xyz2)  # (B, M2, N=3)
	ones1 = tf.ones((fshape1[0],fshape1[1],1))  # (B, M1, 1)
	ones2 = tf.ones((fshape1[0],1,fshape2[1]))  # (B, 1, M2)
	d1 = tf.expand_dims(tf.reduce_sum(xyz1**2, axis=2), axis=2)  # (B, M1, 1)
	d2 = tf.expand_dims(tf.reduce_sum(xyz2**2, axis=2), axis=1)  # (B, 1, M2)
	G12 = tf.matmul(xyz1, tf.transpose(xyz2, perm=[0,2,1]))  # (B, M1, M2)
	edm = tf.matmul(d1, ones2) + tf.matmul(ones1, d2) - 2*G12  # (B, M1, M2)
	cdist12 = tf.reduce_min(edm, axis=2)  # (B, M1)
	cdist21 = tf.reduce_min(edm, axis=1)  # (B, M2)
	return [cdist12, cdist21]

if __name__=='__main__':
	import numpy as np
	import random
	import time
	from tensorflow.python.ops.gradient_checker import compute_gradient
	random.seed(100)
	np.random.seed(100)
	with tf.Session('') as sess:
		xyz1=np.random.randn(BATCH_SIZE*NUM_VIEW,POINTCLOUDSIZE,3).astype('float32')
		xyz2=np.random.randn(BATCH_SIZE*NUM_VIEW,OUTPUTPOINTS,3).astype('float32')
		#with tf.device('/gpu:0'):
		if True:
			inp1=tf.Variable(xyz1)
			inp2=tf.constant(xyz2)
			reta,retb,retc,retd=nn_distance(inp1,inp2)
			loss=tf.reduce_sum(reta)+tf.reduce_sum(retc)
			train=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
		sess.run(tf.initialize_all_variables())
		t0=time.time()
		t1=t0
		best=1e100
		for i in xrange(100):
			trainloss,_=sess.run([loss,train])
			newt=time.time()
			best=min(best,newt-t1)
			print i,trainloss,(newt-t0)/(i+1),best
			t1=newt
		#print sess.run([inp1,retb,inp2,retd])
		#grads=compute_gradient([inp1,inp2],[(16,32,3),(16,32,3)],loss,(1,),[xyz1,xyz2])
		#for i,j in grads:
			#print i.shape,j.shape,np.mean(np.abs(i-j)),np.mean(np.abs(i)),np.mean(np.abs(j))
		#for i in xrange(10):
			#t0=time.time()
			#a,b,c,d=sess.run([reta,retb,retc,retd],feed_dict={inp1:xyz1,inp2:xyz2})
			#print 'time',time.time()-t0
		#print a.shape,b.shape,c.shape,d.shape
		#print a.dtype,b.dtype,c.dtype,d.dtype
		#samples=np.array(random.sample(range(xyz2.shape[1]),100),dtype='int32')
		#dist1=((xyz1[:,samples,None,:]-xyz2[:,None,:,:])**2).sum(axis=-1).min(axis=-1)
		#idx1=((xyz1[:,samples,None,:]-xyz2[:,None,:,:])**2).sum(axis=-1).argmin(axis=-1)
		#print np.abs(dist1-a[:,samples]).max()
		#print np.abs(idx1-b[:,samples]).max()
		#dist2=((xyz2[:,samples,None,:]-xyz1[:,None,:,:])**2).sum(axis=-1).min(axis=-1)
		#idx2=((xyz2[:,samples,None,:]-xyz1[:,None,:,:])**2).sum(axis=-1).argmin(axis=-1)
		#print np.abs(dist2-c[:,samples]).max()
		#print np.abs(idx2-d[:,samples]).max()

