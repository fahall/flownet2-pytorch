import cv2
from glob import glob
import os.path as osp
import numpy as np
import torch
from torch.autograd import Variable
from FlowNet2_src import FlowNet2
from FlowNet2_src import flow_to_image

DATA_DIR = '/data/'
OUT_DIR = osp.join(DATA_DIR, 'flows')
MODEL_HOME = osp.join(DATA_DIR, 'flownet_models')
MODEL_PATH = osp.join(MODEL_HOME,'FlowNet2_checkpoint.pth.tar')
VIDEO_PATTERN = osp.join(DATA_DIR, 'videos/*.mp4')
NETWORK_IMAGE_DIMS = (384, 512)


def get_frame(vidcap, framenum):
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
    result, cv2_im = vidcap.read()
    if result:
        cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        cv2_im = cv2.resize(cv2_im, NETWORK_IMAGE_DIMS)
        return cv2_im
    else:
        return None

def get_video_flow(videofile, step=6):
    cap = cv2.VideoCapture(videofile)
    ims = [get_frame(cap, i*step) for i in range(2)]
    flows = get_flows(ims)
    store_flows(flows, ['test.ppm'])
    

def get_flows(images, network=None):
    ims = prep_ims_for_torch(images)
    if network is None:
        network = get_network()
    pred_flows = network(ims).cpu().data
    return pred_flows

def store_flows(flows, filenames):
    assert len(flows) == len(filenames)
    for i in range(len(flows)):
        store_single_flow(flows[i], filenames[i])

def store_single_flow(flow, path):
    args = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    im = cv2.cvtColor(flow_to_im(flow), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, im, args)

def flow_to_im(flow):
    return flow_to_image(flow.numpy().transpose((1,2,0)))

def prep_ims_for_torch(images):
    ims = np.array([images]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
    ims = torch.from_numpy(ims)
    ims_v = Variable(ims.cuda(), requires_grad=False)
    return ims_v

def get_network():
    network = FlowNet2()
    model_path = MODEL_PATH
    pretrained_dict = torch.load(model_path)['state_dict']
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    network.cuda()
    return network
        
if __name__ == '__main__':
    filepaths = glob(VIDEO_PATTERN)
    test = filepaths[0]
    get_video_flow(test)
