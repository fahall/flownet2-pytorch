import json
import logging
import os.path as osp
import sys
from glob import glob
from os import makedirs

import numpy as np

import cv2
import torch
from FlowNet2_src import FlowNet2, flow_to_image
from torch.autograd import Variable
from tqdm import tqdm

DATA_DIR = '/data/'
OUT_DIR = osp.join(DATA_DIR, 'flows')
MODEL_HOME = osp.join(DATA_DIR, 'flownet_models')
MODEL_PATH = osp.join(MODEL_HOME, 'FlowNet2_checkpoint.pth.tar')
VIDEO_PATTERN = osp.join(DATA_DIR, 'videos/*.mp4')
NETWORK_IMAGE_DIMS = (384, 512)
SCRATCH = 'scratch'
GPU = 0
SCALE = 0.25
BATCH_SIZE = 15
STATUS_FILE = osp.join(DATA_DIR, 'flownet_status.json')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


logger = setup_custom_logger(__name__)


def get_frame(vidcap, framenum):
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
    result, cv2_im = vidcap.read()
    if result:
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im = cv2.resize(cv2_im, NETWORK_IMAGE_DIMS)
        return cv2_im
    else:
        return None


def get_video_flow(videofile, step=6):
    fpath = get_output_conversion_func(videofile)
    cap = cv2.VideoCapture(videofile)
    found_end = False
    strt = 0
    num_processed = 0
    while not found_end:
        frame_nums = range(strt, strt+BATCH_SIZE)
        ims = [get_frame(cap, i*step) for i in frame_nums]
        if not ims:
            logger.info('Stopping. No more images.')
            break
        if ims[-1] is None:
            logger.info('Cleaning up for final run.')
            ims = [i for i in ims if i is not None]
            found_end = True
        num_processed += len(ims)
        flows = get_flows(ims)
        filepaths = [fpath((i+1)*step + strt) for i in range(len(flows))]
        store_flows(flows, filepaths)
        strt += BATCH_SIZE
        if num_processed % 100 == 0:
            logger.info('Finished: ' + str(num_processed))

    return num_processed


def get_subdir(videofile):
    title = osp.splitext(osp.basename(videofile))[0]
    subdir = osp.join(OUT_DIR, title)
    if not osp.exists(subdir):
        makedirs(subdir)
    return subdir


def get_output_conversion_func(videofile):

    def fpath(i): return osp.join(get_subdir(
        videofile), str(i).zfill(6) + '.jpg')
    return fpath


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
    im = cv2.cvtColor(cv2.resize(flow_to_im(flow), (0, 0),
                                 fx=SCALE, fy=SCALE), cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, im, args)


def flow_to_im(flow):
    return flow_to_image(flow.numpy().transpose((1, 2, 0)))


def prep_ims_for_torch(images):
    ims = np.array([[images[i], images[i+1]] for i in range(len(images)-1)])
    ims = ims.transpose((0, 4, 1, 2, 3)).astype(np.float32)
    ims = torch.from_numpy(ims)
    ims_v = Variable(ims.to(DEVICE), requires_grad=False)
    return ims_v


def get_network():
    network = FlowNet2()
    model_path = MODEL_PATH
    pretrained_dict = torch.load(model_path)['state_dict']
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        network = nn.DataParallel(network)

    return network.to(DEVICE)


def store_status(d):
    with open(STATUS_FILE, 'w') as f:
        json.dump(d, f)


def load_status():
    try:
        with open(STATUS_FILE, 'w') as f:
            return json.load(f)
    except:
        return {}


def needs_flows(filepath, finished):
    in_finished = filepath in finished.keys()
    has_directory = osp.exists(get_subdir(filepath))
    return (not in_finished) and (not has_directory)


if __name__ == '__main__':
    filepaths = glob(VIDEO_PATTERN)
    finished = load_status()
    prob_finished = {}
    filepaths = [fp for fp in filepaths if needs_flows(fp, finished)]
    for f in tqdm(filepaths):
        finished[f] = get_video_flow(f)
        store_status(finished)
