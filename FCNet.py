# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:49:20 2017

@author: acnj992
"""


import tensorflow as tf
import numpy as np
import scipy.io as sio

modelPath = 'savedFCNetModel/model.ckpt'
summaryPath = 'FCNetlog/summaryLog'

#sess = tf.Session()

# code tester section
#with tf.Session() as sess:
#    Ww = tf.get_variable('Ww',dtype = tf.float32, shape=[1], initializer = tf.constant(1.2) )
#    init_op = tf.global_variables_initializer()
#    sess.run(init_op)
#
#    print((Ww.name))
##sess.close()
#
#with tf.variable_scope("foo"):
#    v = getVar() # tf.get_variable("v", [1])
#with tf.variable_scope("foo", reuse=True):
#    v1 = getVar() #tf.get_variable("v", [1])
#assert v1 is v
##
#def getVar():
#    return tf.get_variable("v", [1])
#
#assign_op = v1.assign([2])
#sess.run(assign_op)  # or
#print(sess.run(v))
#print(sess.run(v.name))

tf.reset_default_graph()        
init = tf.truncated_normal_initializer(stddev=0.1)

def loadMatlabVar(matlabFileName, varName):
    mat_contents = sio.loadmat(matlabFileName)
    return mat_contents[varName]

def Conv1D(x, nChannels, nameParam = 'conv', kernelSize = 3):
    with tf.name_scope('Conv'):
        # x is in format  [batch, length, channels]
        # create weight var format [inChannel,outChannel, filters]
        inChannels = x.get_shape()[2]
        w = tf.get_variable(name=nameParam+'_w',shape=[kernelSize,inChannels,nChannels],dtype = tf.float32, initializer=init)
        b = tf.get_variable(name=nameParam+'_b',dtype = tf.float32, initializer=tf.constant(0.01, shape=[nChannels], dtype=tf.float32))
        return tf.nn.bias_add(tf.nn.conv1d(value = x,filters=w, stride=1,padding = 'VALID', name = nameParam),b)

def batchNorm(x,nameParam='BN'):
    # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    with tf.name_scope('BN'):
        inChannels = x.get_shape()[2]
        batch_mean, batch_var = tf.nn.moments(x,[0])
        #scale = tf.Variable(tf.ones([inChannels]))
        #offset = tf.Variable(tf.zeros([inChannels]))
        
        offset  = tf.get_variable(name=nameParam+'_offset',shape=[inChannels],dtype = tf.float32, initializer=tf.zeros_initializer() )
        scale  = tf.get_variable(name=nameParam+'_scale',shape=[inChannels],dtype = tf.float32, initializer= tf.ones_initializer() )
        return tf.nn.batch_normalization(x,batch_mean, batch_var,offset,scale,0.01,name=nameParam)

def batchNormWithWeights(x,offset,scale,nameParam='BN'):
    with tf.name_scope('BN'):
        inChannels = x.get_shape()[2]
        batch_mean, batch_var = tf.nn.moments(x,[0])
        return tf.nn.batch_normalization(x,batch_mean,batch_var,offset,scale,0.01,name=nameParam)

def Conv1DWithWeights(x, w, b, nChannels, nameParam = 'conv', kernelSize = 3):
    with tf.name_scope('Conv'):
        # x is in format  [batch, length, channels]    
        return tf.nn.bias_add(tf.nn.conv1d(value = x,filters=w, stride=1,padding = 'VALID', name = nameParam),b)
    
def LeakyReLU(x, nameParam='LeakyReLU'):
    with tf.name_scope('ReLu'):
        alpha = tf.get_variable(name=nameParam+'_w',shape=[1],dtype = tf.float32, initializer=init)
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def LeakyReLUWithWeights(x, alpha, nameParam='LeakyReLU'):
    with tf.name_scope('ReLu'):
        #alpha = tf.get_variable(name=nameParam+'_w',shape=[1],dtype = tf.float32, initializer=init)
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def FullyConnected(x,nodes,nameParam="fc"):
    with tf.name_scope('FC'):
        inNodes = x.get_shape()[1]
        w = tf.get_variable(name=nameParam+'_w',shape=[inNodes,nodes],dtype = tf.float32, initializer=init)
        b = tf.get_variable(name=nameParam+'_b',dtype = tf.float32, initializer=tf.constant(0.01, shape=[nodes], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(x,w),b,name = nameParam)
        return fc

def FullyConnectedWithWeights(x,w,b,nodes,nameParam="fc"):
    with tf.name_scope('FC'):
        fc = tf.nn.bias_add(tf.matmul(x,w),b,name = nameParam)
        return fc


def makeRow(x):
        with tf.name_scope('FS_row'):
            #x = tf.placeholder(tf.float32, shape=[None, 172])
            #x = tf.reshape(x, [-1, 172, 1])
            x1 = Conv1D(x,32.0,'Conv1')
            #bn1 = tf.layers.batch_normalization(x1,name='BN1')
            bn1 = batchNorm(x1,'BN1')
            #activ1 = tf.nn.relu(bn1)
            activ1 = LeakyReLU(bn1,'LReLU1')
            pool1 = tf.layers.max_pooling1d(activ1,pool_size=2,strides=2, name = 'maxPool1')    
            
            c2 = Conv1D(pool1,64.0,'Conv2')
            bn2 = batchNorm(c2,'BN2')
            #activ2 = tf.nn.relu(bn2)
            activ2 = LeakyReLU(bn2,'LReLU2')
            pool2 = tf.layers.max_pooling1d(activ2,pool_size=2,strides=2, name = 'maxPool1')    
            
            c3 = Conv1D(pool2,64.0,'Conv3')
            bn3 = batchNorm(c3,'BN3')
            #activ3 = tf.nn.relu(bn3)
            activ3 = LeakyReLU(bn3,'LReLU3')
            
            c4 = Conv1D(activ3,64,'Conv4')
            c5 = Conv1D(c4,64,'Conv5')
            
            pool3 = tf.layers.max_pooling1d(c5,pool_size=2,strides=2, name = 'maxPool3')    
            
            #flattenSize = pool3.shape[1]*pool3.shape[2]
            flatten = tf.reshape(pool3,[-1,17*64],name='flatten')
            # dense 32        
            fc = FullyConnected(flatten,32,'FC')
            return fc
            
def makeDiffNN(tensor1,tensor2):
    with tf.name_scope('DiffNN'):
        merged = tf.concat([tensor1,tensor2],1,'merge')        
        fc1 = FullyConnected(merged,32,'FC1')
        fc2 = FullyConnected(fc1,32,'FC2')
        predictions = FullyConnected(fc2,2,'predictions')
        return predictions

'''
the function makes a row of FCNet and loads the saved wieghts and all variables
'''
def loadRow(x,graph):
#    with tf.Session() as sess:
#        saver = tf.train.import_meta_graph(modelPath + '.meta')
#        saver.restore(sess,modelPath)
        
#        graph = tf.get_default_graph()  #to get default graph
    with tf.name_scope('FS_row'):
        w = graph.get_tensor_by_name('siameseNet/Conv1_w:0')
        b = graph.get_tensor_by_name('siameseNet/Conv1_b:0')
        wVal = w.eval()
        bVal = b.eval()
        wTensor = tf.get_variable('Conv1_w',initializer=tf.constant(wVal))
        bTensor = tf.get_variable('Conv1_b',initializer=tf.constant(bVal))
        x1 = Conv1DWithWeights(x, wTensor,bTensor,32,'Conv1')
        #bn1 = tf.layers.batch_normal ization(x1,name='BN1')
        
        offset = graph.get_tensor_by_name('siameseNet/BN1_offset:0').eval()
        scale = graph.get_tensor_by_name('siameseNet/BN1_scale:0').eval()
        
        bn1 = batchNormWithWeights(x1,offset,scale,'BN1')
        #activ1 = tf.nn.relu(bn1)
        alpha = graph.get_tensor_by_name('siameseNet/LReLU1_w:0')
        alphaVal = alpha.eval()
        alphaTensor = tf.get_variable('LReLU1_w',initializer=tf.constant(alphaVal))
        activ1 = LeakyReLUWithWeights(bn1,alphaTensor,'LReLU1')
        pool1 = tf.layers.max_pooling1d(activ1,pool_size=2,strides=2, name = 'maxPool1')    
        
        w = graph.get_tensor_by_name('siameseNet/Conv2_w:0').eval()
        b = graph.get_tensor_by_name('siameseNet/Conv2_b:0').eval()
        wTensor = tf.get_variable('Conv2_w',initializer=tf.constant(w))
        bTensor = tf.get_variable('Conv2_b',initializer=tf.constant(b))
        c2 = Conv1DWithWeights(pool1,wTensor,bTensor,64,'Conv2')
        
        offset = graph.get_tensor_by_name('siameseNet/BN2_offset:0').eval()
        scale = graph.get_tensor_by_name('siameseNet/BN2_scale:0').eval()
        bn2 = batchNormWithWeights(c2,offset,scale,'BN2')
        #activ2 = tf.nn.relu(bn2)
        alpha = graph.get_tensor_by_name('siameseNet/LReLU2_w:0').eval()
        alphaTensor = tf.get_variable('LReLU2_w',initializer=tf.constant(alpha))
        activ2 = LeakyReLUWithWeights(bn2,alphaTensor,'LReLU2')
        pool2 = tf.layers.max_pooling1d(activ2,pool_size=2,strides=2, name = 'maxPool1')    
        
        w = graph.get_tensor_by_name('siameseNet/Conv3_w:0').eval()
        b = graph.get_tensor_by_name('siameseNet/Conv3_b:0').eval()
        wTensor = tf.get_variable('Conv3_w',initializer=tf.constant(w))
        bTensor = tf.get_variable('Conv3_b',initializer=tf.constant(b))
        c3 = Conv1DWithWeights(pool2,wTensor,bTensor,64,'Conv3')
        
        offset = graph.get_tensor_by_name('siameseNet/BN3_offset:0').eval()
        scale = graph.get_tensor_by_name('siameseNet/BN3_scale:0').eval()
        bn3 = batchNormWithWeights(c3,offset,scale,'BN3')
        #activ3 = tf.nn.relu(bn3)
        alpha = graph.get_tensor_by_name('siameseNet/LReLU3_w:0').eval()
        alphaTensor = tf.get_variable('LReLU3_w',initializer=tf.constant(alpha))
        activ3 = LeakyReLUWithWeights(bn3,alphaTensor,'LReLU3')
        
        w = graph.get_tensor_by_name('siameseNet/Conv4_w:0').eval()
        b = graph.get_tensor_by_name('siameseNet/Conv4_b:0').eval()
        wTensor = tf.get_variable('Conv4_w',initializer=tf.constant(w))
        bTensor = tf.get_variable('Conv4_b',initializer=tf.constant(b))
        c4 = Conv1DWithWeights(activ3,wTensor,bTensor,64,'Conv4')
        
        w = graph.get_tensor_by_name('siameseNet/Conv5_w:0').eval()
        b = graph.get_tensor_by_name('siameseNet/Conv5_b:0').eval()
        wTensor = tf.get_variable('Conv5_w',initializer=tf.constant(w))
        bTensor = tf.get_variable('Conv5_b',initializer=tf.constant(b))
        c5 = Conv1DWithWeights(c4,wTensor,bTensor,64,'Conv5')
        
        pool3 = tf.layers.max_pooling1d(c5,pool_size=2,strides=2, name = 'maxPool3')    
        
        #flattenSize = pool3.shape[1]*pool3.shape[2]
        flatten = tf.reshape(pool3,[-1,17*64],name='flatten')
        # dense 32        
        w = graph.get_tensor_by_name('siameseNet/FC_w:0').eval()
        b = graph.get_tensor_by_name('siameseNet/FC_b:0').eval()
        wTensor = tf.get_variable('FC_w',initializer=tf.constant(w))
        bTensor = tf.get_variable('FC_b',initializer=tf.constant(b))
        fc = FullyConnectedWithWeights(flatten,wTensor,bTensor,32,'FC')
        return fc

def loadDiffNN(tensor1,tensor2,modelPath):
    with tf.name_scope('DiffNN'):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(modelPath + '.meta')
            saver.restore(sess,modelPath)
            graph = tf.get_default_graph()  #to get default graph
            
            merged = tf.concat([tensor1,tensor2],1,'merge')        
            
            w = graph.get_tensor_by_name('FC1_w:0').eval()
            b = graph.get_tensor_by_name('FC1_b:0').eval()
            wTensor = tf.get_variable('FC1_w',initializer=tf.constant(w))
            bTensor = tf.get_variable('FC1_b',initializer=tf.constant(b))
            fc1 = FullyConnectedWithWeights(merged,wTensor,bTensor,32,'FC1')
            
            w = graph.get_tensor_by_name('FC2_w:0').eval()
            b = graph.get_tensor_by_name('FC2_b:0').eval()
            wTensor = tf.get_variable('FC2_w',initializer=tf.constant(w))
            bTensor = tf.get_variable('FC2_b',initializer=tf.constant(b))
            fc2 = FullyConnectedWithWeights(fc1,wTensor,bTensor,32,'FC2')
            
            w = graph.get_tensor_by_name('predictions_w:0').eval()
            b = graph.get_tensor_by_name('predictions_b:0').eval()
            wTensor = tf.get_variable('predictions_w',initializer=tf.constant(w))
            bTensor = tf.get_variable('predictions_b',initializer=tf.constant(b))
            predictions = FullyConnectedWithWeights(fc2,wTensor,bTensor,2,'predictions')
        return predictions

'''
The function builds the FCNet model layers by layer and 
loads the weights for each layer from the saved model provided in modelPath
'''
def evaluateSavedFCNet():
    signalSrc = loadMatlabVar('FCNetTrainData.mat','signalSrc')
    signalDest = loadMatlabVar('FCNetTrainData.mat','signalDest')
    signalLabel = loadMatlabVar('FCNetTrainData.mat','signalLabel')
    
    x1 = tf.placeholder(tf.float32, shape=[None, 172,1])
    y = tf.placeholder(tf.float32, shape=[None, 2])
    #x1 = tf.reshape(x1, [-1, 172, 1])
    x2 = tf.placeholder(tf.float32, shape=[None, 172,1])
    #x2 = tf.reshape(x2, [-1, 172, 1])
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(modelPath + '.meta')
        saver.restore(sess,modelPath)
        
        graph = tf.get_default_graph()  #to get default graph

        with tf.variable_scope("siameseNet") as scope:
            p1 = loadRow(x1,graph)
            scope.reuse_variables()
            p2 = loadRow(x2,graph)
    
    predictions = loadDiffNN(p1,p2,modelPath)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predictions))
    optimizer = tf.train.AdamOptimizer()  #optimizer = tf.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
    
    trainLabelHV = np.array( list(map(lambda x: (0,1) if x==1 else (1,0),signalLabel)))
# soem hyper params
    nSamples = trainLabelHV.shape[0]
    nEpochs = 2
    batch_size = 100
    nBatches = int(nSamples / batch_size)

    trainCol1 = signalSrc.reshape(signalSrc.shape[0],signalSrc.shape[1],1)
    trainCol2 = signalDest.reshape(signalDest.shape[0],signalDest.shape[1],1)

    with tf.Session() as sess:
      # init all vars  
      sess.run(tf.global_variables_initializer())
      #saver = tf.train.import_meta_graph(modelPath + '.meta')
      #saver.restore(sess,tf.train.latest_checkpoint('savedModel/'))
      
      
      for e in range(nEpochs):
          file = open("results e{}.txt".format(e),"w") 
          loss=0
          train_accuracy =0
          
          for batchNumber in range (0,nBatches):
              #bid*batch_size:(bid+1)*batch_size
              startInd = batchNumber * batch_size
              endInd = (batchNumber + 1) * batch_size
              sess.run(train_step,feed_dict={x1: trainCol1[startInd:endInd,:,:], 
                                        x2: trainCol2[startInd:endInd,:,:], 
                                        y: trainLabelHV[startInd:endInd,:]
                                        })
#              train_step.run(feed_dict={x1: trainCol1[startInd:endInd,:], 
#                                        x2: trainCol2[startInd:endInd,:], 
#                                        y: trainLabelHV[startInd:endInd,:]
#                                        })
              crossLoss = cross_entropy.eval({x1: trainCol1[startInd:endInd,:], 
                                        x2: trainCol2[startInd:endInd,:], 
                                        y: trainLabelHV[startInd:endInd,:]
                                        })     
              loss += crossLoss
              
              acc = accuracy.eval(feed_dict={x1: trainCol1[startInd:endInd,:], 
                                        x2: trainCol2[startInd:endInd,:], 
                                        y: trainLabelHV[startInd:endInd,:]
                                        })
              train_accuracy += acc
              file.write("e {}, batch id {} accuracy {} loss {} \n".format(e,batchNumber,acc,crossLoss))
        
          print('epoch %d, loss %g training accuracy %g' % (e, loss/nBatches,train_accuracy/nBatches))
          file.close()

def evaluateSavedModel(signalSrc, signalDest,signalLabel,metaFile):
    
    x1 = tf.placeholder(tf.float32, shape=[None, 172,1])
    y = tf.placeholder(tf.float32, shape=[None, 2])
    #x1 = tf.reshape(x1, [-1, 172, 1])
    x2 = tf.placeholder(tf.float32, shape=[None, 172,1])
    #x2 = tf.reshape(x2, [-1, 172, 1])
    
    with tf.variable_scope("siameseNet") as scope:
        p1 = makeRow(x1)
        scope.reuse_variables()
        p2 = makeRow(x2)
    
    predictions = makeDiffNN(p1,p2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predictions))
    optimizer = tf.train.AdamOptimizer()  #optimizer = tf.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(cross_entropy)
    tf.summary.scalar("cost",cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
    with tf.name_scope("sumamries"):
        
        tf.summary.scalar("accuracy",accuracy)
        tf.summary.scalar('Loss',cross_entropy)
        tf.summary.histogram('histogram_loss',cross_entropy)
        tf.summary.histogram('histogram_accuracy',accuracy)
        summary_op = tf.summary.merge_all()
    
    
    trainLabelHV = np.array( list(map(lambda x: (0,1) if x==1 else (1,0),signalLabel)))
# soem hyper params
    nSamples = trainLabelHV.shape[0]
    nEpochs = 4
    batch_size = 100
    nBatches = int(nSamples / batch_size)

    trainCol1 = signalSrc.reshape(signalSrc.shape[0],signalSrc.shape[1],1)
    trainCol2 = signalDest.reshape(signalDest.shape[0],signalDest.shape[1],1)

    with tf.Session() as sess:
      # init all vars  
      sess.run(tf.global_variables_initializer())
      #saver = tf.train.import_meta_graph(modelPath + '.meta')
      #saver.restore(sess,tf.train.latest_checkpoint('savedModel/'))
      
      
      saver = tf.train.Saver()
      saver.restore(sess,modelPath)

      
      # graph = tf.get_default_graph()  #to get default graph
      # nodeList = [n.name for n in tf.get_default_graph().as_graph_def().node]
      for e in range(nEpochs):
          file = open("results e{}.txt".format(e),"w") 
          loss=0
          train_accuracy =0
          
          for batchNumber in range (0,nBatches):
              #bid*batch_size:(bid+1)*batch_size
              startInd = batchNumber * batch_size
              endInd = (batchNumber + 1) * batch_size
    #          sess.run(train_step,feed_dict={x1: trainCol1[startInd:endInd,:,:], 
    #                                    x2: trainCol2[startInd:endInd,:,:], 
    #                                    y: trainLabelHV[startInd:endInd,:]
    #                                    })
#              train_step.run(feed_dict={x1: trainCol1[startInd:endInd,:], 
#                                        x2: trainCol2[startInd:endInd,:], 
#                                        y: trainLabelHV[startInd:endInd,:]
#                                        })
              crossLoss = cross_entropy.eval({x1: trainCol1[startInd:endInd,:], 
                                        x2: trainCol2[startInd:endInd,:], 
                                        y: trainLabelHV[startInd:endInd,:]
                                        })     
              loss += crossLoss
              
              acc = accuracy.eval(feed_dict={x1: trainCol1[startInd:endInd,:], 
                                        x2: trainCol2[startInd:endInd,:], 
                                        y: trainLabelHV[startInd:endInd,:]
                                        })
              train_accuracy += acc
              file.write("e {}, batch id {} accuracy {} loss {} \n".format(e,batchNumber,acc,crossLoss))
        
          print('epoch %d, loss %g training accuracy %g' % (e, loss/nBatches,train_accuracy/nBatches))
          file.close()
   
    
    
def trainModel(nEpochs=10):    
#def train():    
    x1 = tf.placeholder(tf.float32, shape=[None, 172,1])
    y = tf.placeholder(tf.float32, shape=[None, 2])
    #x1 = tf.reshape(x1, [-1, 172, 1])
    x2 = tf.placeholder(tf.float32, shape=[None, 172,1])
    #x2 = tf.reshape(x2, [-1, 172, 1])
    with tf.name_scope('siameseNet_scope'):
        with tf.variable_scope("siameseNet") as scope:
            p1 = makeRow(x1)
            scope.reuse_variables()
            p2 = makeRow(x2)
    with tf.name_scope('DiffNet_scope'):
        with tf.variable_scope("DiffNNet") as scope:
            predictions = makeDiffNN(p1,p2) 
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predictions),name='crossEnt')
    optimizer = tf.train.AdamOptimizer()  #optimizer = tf.train.AdamOptimizer(1e-4)
    with tf.name_scope('train'):
        train_step = optimizer.minimize(cross_entropy,name='train')
    #tf.summary.scalar("cost",cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
    with tf.name_scope("sumamries"):
        
        tf.summary.scalar("accuracy",accuracy)
        tf.summary.scalar('Loss',cross_entropy)
        tf.summary.histogram('histogram_loss',cross_entropy)
        tf.summary.histogram('histogram_accuracy',accuracy)
        summary_op = tf.summary.merge_all()
    # model created with loss 
    
    # optional - load data
    signalSrc = loadMatlabVar('FCNetTrainData.mat','signalSrc')
    signalDest = loadMatlabVar('FCNetTrainData.mat','signalDest')
    signalLabel = loadMatlabVar('FCNetTrainData.mat','signalLabel')
    
    # process the data now
    trainLabelHV = np.array( list(map(lambda x: (0,1) if x==1 else (1,0),signalLabel)))
    # soem hyper params
    nSamples = trainLabelHV.shape[0]
    #nEpochs = 10
    batch_size = 100
    nBatches = int(nSamples / batch_size)
    
    trainCol1 = signalSrc.reshape(signalSrc.shape[0],signalSrc.shape[1],1)
    trainCol2 = signalDest.reshape(signalDest.shape[0],signalDest.shape[1],1)
    
    with tf.Session() as sess:
      # init all vars  
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      
      writer = tf.summary.FileWriter(summaryPath,graph=tf.get_default_graph())
      for e in range(nEpochs):
          file = open("results e{}.txt".format(e),"w") 
          loss=0
          train_accuracy =0
          
          for batchNumber in range (0,nBatches):
              #bid*batch_size:(bid+1)*batch_size
              startInd = batchNumber * batch_size
              endInd = (batchNumber + 1) * batch_size
        
              _, crossLoss,acc,summary = sess.run([train_step,cross_entropy,accuracy,summary_op], feed_dict={x1: trainCol1[startInd:endInd,:], 
                                        x2: trainCol2[startInd:endInd,:], 
                                        y: trainLabelHV[startInd:endInd,:]
                                        })
              loss += crossLoss
              train_accuracy += acc
              file.write("e {}, batch id {} accuracy {} loss {} \n".format(e,batchNumber,acc,crossLoss))
              
          print('epoch %d of %d, loss %g training accuracy %g' % (e, nEpochs ,loss/nBatches,train_accuracy/nBatches))
          writer.add_summary(summary,e)
          file.close()
      save_path = saver.save(sess, modelPath)
      print("Model saved in file: %s" % save_path)
              
        
        #  print('test accuracy %g' % accuracy.eval(feed_dict={
        #      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))