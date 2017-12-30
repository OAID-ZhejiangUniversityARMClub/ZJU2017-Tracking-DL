import numpy as np
import caffe
from caffe import layers as L, params as P
import caffe.proto.caffe_pb2 as caffe_pb2
import Weight_Bias_CP
import prune
import math

goturn_net = caffe.Net('goturnDeploy.prototxt', 'goturn_iter_580000.caffemodel', caffe.TEST)
goturn_netnew = caffe.Net('goturnDeploy.prototxt', caffe.TEST)
#print goturn_net.top_names.keys()

layer_with_weights_name = []
for layer_name in goturn_net.top_names.keys():
    if layer_name.startswith('conv') or layer_name.endswith('/squeeze1x1') or layer_name.endswith('/expand1x1') \
                                                                           or layer_name.endswith('/expand3x3') \
                                                                           or layer_name.endswith('/squeeze1x1_p') \
                                                                           or layer_name.endswith('/expand1x1_p') \
                                                                           or layer_name.endswith('/expand3x3_p'):
        layer_with_weights_name.append(layer_name)
print layer_with_weights_name

'''
print goturn_net.params[layer_with_weights_name[1]][0].data.flatten()

leng = len(layer_with_weights_name)
prune.draw_hist_weight(goturn_net, layer_with_weights_name[:28])
prune.draw_hist_weight(goturn_net, layer_with_weights_name[28:])
'''

#prune.analyze_param(goturn_net, layer_with_weights_name)


for layer_name in layer_with_weights_name:
    count = 0
    weight = goturn_net.params[layer_name][0].data
    weightnew=[]
    for i in weight.flat:
        if abs(i) < 0.01:
            count += 1
            i = 0
        weightnew.append(i)
    goturn_netnew.params[layer_name][0].data.flat = weightnew
    print count

goturn_netnew.params['fc6'][0].data.flat = goturn_net.params['fc6'][0].data.flat
goturn_netnew.params['fc7'][0].data.flat = goturn_net.params['fc7'][0].data.flat
goturn_netnew.save('goturn_prune580000.caffemodel')
