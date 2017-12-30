import numpy as np
import caffe
from caffe import layers as L, params as P
import caffe.proto.caffe_pb2 as caffe_pb2
import Weight_Bias_CP

BATCH_SIZE = 10
caffe_root = '/home/wj/caffe-master/'

caffe_net = caffe.Net(caffe_root+'models/bvlc_reference_caffenet/deploy.prototxt',
                    caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', caffe.TEST)
#goturn_net = caffe.Net('/home/wj/software/Goturn_Training/GOTURN_Training-master/goturnTrain.prototxt', caffe.TEST)
#model = caffe_pb2.NetParameter()
#with open(caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', 'rb') as f:
#    model.ParseFromString(f.read())


num_of_conv = ','.join(caffe_net.top_names.keys()).count('conv')
Norm_num = ['norm'+str(i) for i in range(1, num_of_conv+1)]
Norm_flag = [1*i in caffe_net.top_names for i in Norm_num]
STRIDE = [11, 5, 3, 3, 3]

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2
'''
names = locals()
index=1
names['conv1%s'%index] = 1
n = caffe.NetSpec()
n.data1, n.data2, n.label = L.HDF5Data(batch_size=11, source="trainDatasets.txt", shuffle='true', ntop=3)
n.tops['conv1'+str(index)] = L.Convolution(n.data1, kernerl_size=5, num_output=96, weight_filler=dict(tpye='gaussian', std=0.01))
'''



def conv_relu_norm_pos(index, bottom, n):
    n.tops['conv1'+str(index)] = L.Convolution(bottom, kernel_size=caffe_net.params['conv'+str(index)][0].data.shape[2],
                                               num_output=caffe_net.params['conv'+str(index)][0].data.shape[0],
                                               stride=STRIDE[index-1], param=frozen_param,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0), pad=2)
    n.tops['relu1'+str(index)] = L.ReLU(n.tops['conv1'+str(index)], in_place=True)
    n.tops['pool1'+str(index)] = L.Pooling(n.tops['relu1'+str(index)], kernel_size=2, stride=2, pool=P.Pooling.MAX)
    if Norm_flag[index-1]:
        n.tops['norm1'+str(index)] = L.LRN(n.tops['pool1'+str(index)], lrn_param=dict(local_size=5, alpha=0.0001, beta=0.75))
        return n.tops['norm1'+str(index)]
    return n.tops['pool1'+str(index)]


def conv_relu_norm_neg(index, bottom, n):
    n.tops['conv2'+str(index)] = L.Convolution(bottom, kernel_size=caffe_net.params['conv'+str(index)][0].data.shape[2],
                                               num_output=caffe_net.params['conv'+str(index)][0].data.shape[0],
                                               stride=4, param=frozen_param,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0), pad=2)
    n.tops['relu2'+str(index)] = L.ReLU(n.tops['conv2'+str(index)], in_place=True)
    n.tops['pool2'+str(index)] = L.Pooling(n.tops['relu2'+str(index)], kernel_size=2, stride=2, pool=P.Pooling.MAX)
    if Norm_flag[index-1]:
        n.tops['norm2'+str(index)] = L.LRN(n.tops['pool2'+str(index)], lrn_param=dict(local_size=5, alpha=0.0001, beta=0.75))
        return n.tops['norm2'+str(index)]
    return n.tops['pool2'+str(index)]

def prototxt_generation(HDF, caffe_net, batch_size=BATCH_SIZE):
    n = caffe.NetSpec()
    #n.data1, n.data2, n.label = L.HDF5Data(batch_size=batch_size, source=HDF, ntop=3, shuffle=True,
    #                                       include=dict(phase=caffe.TRAIN))
    #n.data1, n.data2, n.label = L.HDF5Data(batch_size=batch_size, source=HDF, ntop=3, shuffle=True,
    #                                       include=dict(phase=caffe.TEST))
    n.data1, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=HDF, transform_param=dict(scale=1. / 255), ntop=2)
    #n.data1, n.label = L.Input(shape=dict())
    conv_num = (','.join(caffe_net.top_names.keys())).count('conv')
    bottom1 = n.data1
    bottom2 = n.data1
    for index in range(conv_num):
        bottom1 = conv_relu_norm_pos(index+1, bottom1, n)
    for index in range(conv_num):
        bottom2 = conv_relu_norm_neg(index+1, bottom2, n)
    n.concat1 = L.Concat(bottom1, bottom2)
    n.tops['fc'+str(conv_num+1)+'_new'] = L.InnerProduct(n.concat1,
                                                  num_output=caffe_net.params['fc'+str(conv_num+1)][0].data.shape[0],
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant', value=0),
                                                  param=learned_param)
    n.tops['relu'+str(conv_num+1)] = L.ReLU(n.tops['fc'+str(conv_num+1)+'_new'], in_place=True)
    n.tops['dropout'+str(conv_num+1)] = L.Dropout(n.tops['relu'+str(conv_num+1)], dropout_ratio=0.5)

    n.tops['fc'+str(conv_num+2)+'_new'] = L.InnerProduct(n.tops['dropout'+str(conv_num+1)],
                                                  num_output=caffe_net.params['fc' + str(conv_num + 2)][0].data.shape[0],
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant', value=0),
                                                  param=learned_param)
    n.tops['relu'+str(conv_num+2)] = L.ReLU(n.tops['fc'+str(conv_num+2)+'_new'], in_place=True)
    n.tops['dropout'+str(conv_num+2)] = L.Dropout(n.tops['relu'+str(conv_num+2)], dropout_ratio=0.5)

    n.tops['fc'+str(conv_num+3)+'_new'] = L.InnerProduct(n.tops['dropout'+str(conv_num+2)],
                                                  num_output=4,
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant', value=0),
                                                  param=learned_param)
    #n.out = L.Power(n.tops['fc'+str(conv_num+3)+'_new'], power_param=dict(power=1, scale=10, shift=0))
    #n.loss = L.EuclideanLoss(n.out, n.label)
    n.loss = L.SoftmaxWithLoss(n.tops['fc'+str(conv_num+3)+'_new'], n.label)
    #n.loss = L.SoftmaxWithLoss(n.out, n.label)
    return n.to_proto()

'''
with open('Model/test_model.prototxt', 'w') as f:
    #f.write(str(prototxt_generation("trainDatasets.txt", caffe_net)))
    f.write(str(prototxt_generation('/home/wj/caffe-master/examples/mnist/mnist_train_lmdb', caffe_net)))
'''
Weight_Bias_CP.Solver_generation('Model/test_model.prototxt')
solver = caffe.get_solver('Model/solver.prototxt')
weights = 'Model/test_model.caffemodel'
solver.net.copy_from(weights)

goturn_net = caffe.Net('Model/test_model.prototxt', caffe.TEST)
layer_with_weights_name = []
for layer_name in goturn_net.top_names.keys():
    if layer_name.startswith('conv'):
        layer_with_weights_name.append(layer_name)

Weight_Bias_CP.Weight_Bais_CP(goturn_net, caffe_net, layer_with_weights_name)
solver_path = Weight_Bias_CP.Solver_generation('Model/test_model.prototxt')
solver = caffe.get_solver(solver_path)

weights = 'Model/test_model.caffemodel'
solver.net.copy_from(weights)

solver.solve()
