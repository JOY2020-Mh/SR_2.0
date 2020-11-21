import os
os.environ['GLOG_minloglevel'] = '0'
import caffe
import numpy as np
import google.protobuf
import caffe.proto.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
from collections import OrderedDict
import torch

torch_model = '1121_Visstyle_Steven_dataset_only_random_crop_multi_GPU_Net_epoch_1000.pkl'
save_path = '1121_Visstyle_Steven_dataset_only_random_crop_multi_GPU_Net_epoch_1000.model'
#torch_model = '1119_visdon_HR_downsampling_add_padding_Net_epoch_340.pkl'
#save_path = '1119_visdon_HR_downsampling_add_padding_Net_epoch_340.model'

prototxt="caffe.prototxt"
caffemodel="caffe2.caffemodel"
def _get_ksize(param):
    if param.kernel_h > 0 and param.kernel_w > 0:
        kernel = np.asarray([param.kernel_h, param.kernel_w])
    else:
        kernel = np.asarray([param.kernel_size, param.kernel_size])
    return kernel.reshape(-1)


def _get_stride(param):
    if param.stride_h > 0 and param.stride_w > 0:
        stride = np.asarray([param.stride_h, param.stride_w])
    else:
        stride = np.asarray([param.stride, param.stride])
    stride = stride.reshape(-1)
    if (len(stride) == 0):
        stride = np.asarray([1, 1])
    return stride


def _get_pad(param):
    if param.pad_h > 0 and param.pad_w > 0:
        pad = np.asarray([param.pad_h, param.pad_h, param.pad_w, param.pad_w])
    else:
        pad = np.asarray([param.pad, param.pad, param.pad, param.pad])
    pad = pad.reshape(-1)
    if len(pad) == 0:
        pad = np.asarray([0, 0, 0, 0])
    return pad
    
# load torch model
from network import Net as net
model = net(num_channels=1, scale_factor=4, d=32, s=5, m=1)
# load torch model
original_model_dict = torch.load(torch_model, map_location=torch.device('cpu'))
print(original_model_dict.keys())
print('torch_model_dict.keys()')

torch_model_dict = OrderedDict()
for k, v in original_model_dict.items():
    name = k[7:] # remove `module.`
    torch_model_dict[name] = v
    # load params

keys = [_ for _ in torch_model_dict.keys()]
torch_weights = []
i = 0
print(keys)
while i < len(keys):

    weight = torch_model_dict[keys[i]].cpu().numpy()
    bias = torch_model_dict[keys[i+1]].cpu().numpy()
    print(keys[i], keys[i+1], weight.shape, bias.shape)
    i += 2
    torch_weights.append((weight, bias))
    if i < len(keys) and 'act.weight' in keys[i]:
        act = torch_model_dict[keys[i]].cpu().numpy()
        torch_weights.append(act)
        print(keys[i], act.shape)
        i += 1

# caffe model
#caffe.set_device(0)
#caffe.set_mode_gpu()

net_prototext = caffe_pb2.NetParameter()
text_format.Merge(open(prototxt).read(), net_prototext)

net = caffe.Net(prototxt, caffe.TEST)
weights = net.params

dims = net_prototext.layer[0].input_param.shape[0]
input_shape = [1, dims.dim[1],dims.dim[2],dims.dim[3]]
print(input_shape)

pre_channel = input_shape[0]

print('\n\ntorch2caffe\n\n')
write_result = OrderedDict()
torch_i = 0
for layer in net_prototext.layer[1:]:
    name = layer.name
    type = layer.type
    print(type)
    if type == 'Convolution':
        result = {
                'type': layer.type,
                'name': name,
                'weights': [],
                'param': {}
                }
        param = layer.convolution_param
        group = param.group
        num_outputs = param.num_output
        kernel_size = _get_ksize(param)
        stride = _get_stride(param)
        padding = _get_pad(param)
        use_bias = 'true' if param.bias_term else 'false'

        assert (group == 1)
        assert (use_bias == 'true')
        assert (stride[0] == stride[1])
        assert (np.max(padding) == np.min(padding))
        assert (kernel_size[0] == kernel_size[1])

        stride = stride[0]
        kernel_size = kernel_size[0]
        padding = padding[0]
        params = np.asarray([num_outputs, kernel_size, stride, padding], dtype=np.int32)
        d1, d2 = weights[name]
        torch_w, torch_b = torch_weights[torch_i]
        d1.data[...] = torch_w
        d2.data[...] = torch_b
        print(d1.data.shape, torch_w.shape)
        print(d2.data.shape, torch_b.shape)
        torch_i += 1

        result['weights'] = [d.data.astype(np.float32) for d in weights[name]]
        result['param'] = params
        pre_channel = num_outputs

        write_result[name] = result

    elif type == 'PReLU':
        result = {
            'type': layer.type,
            'name': name,
            'weights': [],
            'param': {}
        }
        param = layer.prelu_param
        assert(not param.channel_shared)
        params = np.asarray([pre_channel], dtype=np.int32)
        result['param'] = params

        (d1, ) = weights[name]
        torch_act = torch_weights[torch_i]
        print(d1.data.shape, torch_act.shape)
        d1.data[...] = torch_act
        torch_i += 1

        result['weights'] = [d.data.astype(np.float32) for d in weights[name]]
        write_result[name] = result
    elif type == 'Deconvolution':
        result = {
            'type': layer.type,
            'name': name,
            'weights': [],
            'param': {}
        }
        param = layer.convolution_param
        group = param.group
        num_outputs = param.num_output
        kernel_size = _get_ksize(param)
        stride = _get_stride(param)
        padding = _get_pad(param)
        use_bias = 'true' if param.bias_term else 'false'

        assert (group == 1)
        assert (use_bias == 'true')
        assert (stride[0] == stride[1])
        assert (np.max(padding) == np.min(padding))
        assert (kernel_size[0] == kernel_size[1])

        stride = stride[0]
        kernel_size = kernel_size[0]
        padding = padding[0]

        params = np.asarray([num_outputs, kernel_size, stride, padding], dtype=np.int32)
        result['weights'] = [d.data.astype(np.float32) for d in weights[name]]
        result['param'] = params
        pre_channel = num_outputs

        result['param'] = params

        d1, d2 = weights[name]
        torch_w, torch_b = torch_weights[torch_i]
        d1.data[...] = torch_w
        d2.data[...] = torch_b
        print(d1.data.shape, torch_w.shape)
        print(d2.data.shape, torch_b.shape)
        torch_i += 1

        result['weights'] = [d.data.astype(np.float32) for d in weights[name]]
        write_result[name] = result

print(write_result.keys())

net.save(caffemodel)


input_shape = np.asarray(input_shape, dtype=np.int32)

print('\n\ncaffe2THIS\n\n')
with open(save_path,'wb') as f:
    for _, result in write_result.items():
        type = result['type']
        print(type)
        if (type == 'Convolution'):
            params = result['param']
            params = np.asarray(params, dtype=np.int32)
            print(params)
            weights = result['weights']
            for d in weights:
                d.astype(np.float32).tofile(f)
                print(d.shape)
        elif (type == 'PReLU'):
            params = result['param']
            params = np.asarray(params, dtype=np.int32)
            weights = result['weights']
            print(params)
            for d in weights:
                d.astype(np.float32).tofile(f)
                print(d.shape)
        elif (type == 'Deconvolution'):
            params = result['param']
            params = np.asarray(params, dtype=np.int32)
            print(params)
            weights = result['weights']
            for d in weights:
                d.astype(np.float32).tofile(f)
                print(d.shape)
