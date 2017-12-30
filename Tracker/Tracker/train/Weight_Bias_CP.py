
import caffe
import tempfile
from caffe.proto import caffe_pb2

def Weight_Bais_CP (goturn_net, caffe_net, layer_name_table):
    for layer_name in layer_name_table:
        if layer_name.endswith('_p'):
            layer_name_caffenet = layer_name[:-2]
        else:
            layer_name_caffenet = layer_name
        weight_goturn = goturn_net.params[layer_name][0].data
        weight_caffenet = caffe_net.params[layer_name_caffenet][0].data
        #bias_goturn = goturn_net.params[layer_name][1].data
        #bias_caffenet = caffe_net.params[layer_name_caffenet][1].data
        #bias_goturn[...] = bias_caffenet
        weight_goturn.flat = weight_caffenet.flat

    goturn_net.save('squeezenetCP/squeezegoturn.caffemodel')

'''
    params = ['fc6', 'fc7', 'fc8']
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

net_full_conv = caffe.Net(caffe_root+'net_surgery/bvlc_caffenet_full_conv.prototxt',
                          '/home/wj/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                          caffe.TEST)
params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)


for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat
    conv_params[pr_conv][1][...]= fc_params[pr][1]

#net_full_conv.save(caffe_root+'net_surgery/bvlc_caffenet_full_conv.caffemodel')

'''

def Solver_generation(train_net, test_net=None, test_interval=500, test_iter=100, max_iter=100000, base_lr=0.001,
                      momentum=0.9, weight_decay=0.9, lr_policy='step', stepsize=10000, gamma=0.1, display=1000,
                      snapshot=50000, prefix='Model/custom_net', type='SGD'):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net
    if test_net is not None:
        s.test_net.append(test_net)
        s.test_interval = test_interval
        s.test_iter.append(test_iter)
    s.max_iter = max_iter

    s.base_lr = base_lr
    s.momentum =momentum
    s.weight_decay = weight_decay
    if lr_policy is 'step':
        s.lr_policy = lr_policy
        s.stepsize = stepsize
    else:
        s.lr_policy = lr_policy  # fixed inv...
    s.gamma = gamma
    s.display = display
    s.snapshot = snapshot
    s.snapshot_prefix = prefix
    s.type = type
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    with open('Model/solver.prototxt', 'w') as f:
        f.write(str(s))







