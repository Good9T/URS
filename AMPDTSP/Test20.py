DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "")  # for problem_def
sys.path.insert(0, "../..")  # for utils

import logging

from utils import create_logger
from Tester import Tester

env_params = {
    'customer_size': 20,
    'linehaul_customer_size': 10,
    'int_min': 0,
    'int_max': 1000*1000,
    'scaler': 1000*1000,
    'asym_bias': 0,
    'capacity':30,
    'demand_low':1,
    'demand_high':10,
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**(1/2),
    'encoder_layer_num': 5,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'argmax',
    'use_sinkhorn': True,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/train20',  # directory path of pre-trained model and log files saved.
        'epoch': 2000,  # epoch version of pre-trained model to load.
    },
    'test_episodes':1000,
    'test_batch_size': 100,
    'aug_enable': True,
    'aug_factor': 10,
    'aug_batch_size': 50,
}
if tester_params['aug_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test20',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    tester.run()


def _set_debug_mode():
    tester_params['aug_factor'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()