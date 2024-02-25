import os
import sys
sys.path.append('/DeepXi/')
from deepxi.args import get_args
from deepxi.model import DeepXi
from deepxi.prelim import Prelim
from deepxi.se_batch import Batch
import deepxi.utils as utils
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
f_s= 16000
T_d = 32
T_s = 16
MIN_SNR=-10
MAX_SNR=20
SNR_INTER=1
# test_x, test_x_len, _, test_x_base_names = []
N_d = int(f_s*T_d*0.001) # window duration (samples).
N_s = int(f_s*T_s*0.001) # window shift (samples).
K = int(pow(2, np.ceil(np.log2(N_d)))) # number of DFT components.
# test_x, test_x_len, _, test_x_base_names = Batch('test')
deepxi = DeepXi(
    gpu=0,
		N_d = N_d,
		N_s = N_d,
		K= K,
    ver="mhanet-1.1c",
    test_epoch=200,
    train=False,
    infer=True,
    test=False,
    spect_dist=False,
    prelim=False,
    verbose=False,
    network_type="MHANetV3",
    inp_tgt_type="MagXi",
    sd_snr_levels=[-5, 0, 5, 10, 15],
    mbatch_size=8,
    sample_size=1000,
    max_epochs=200,
    resume_epoch=0,
    save_model=True,
    log_iter=True,
    eval_example=True,
    val_flag=True,
    reset_inp_tgt=False,
    reset_sample=False,
    out_type="y",
    gain="mmse-lsa",
    model_path="../model",
    set_path="../set",
    log_path="../log",
    sample_dir="../data",
    test_x_path="../test",
    test_s_path="../set/test_clean_speech",
    test_d_path="../set/test_noise",
    out_path="../out",
    saved_data_path=None,
    min_snr=-10,
    max_snr=20,
    snr_inter=1,
    f_s=16000,
    T_d=32,
    T_s=16,
    n_filters=None,
    d_in=None,
    d_out=None,
    d_model=256,
    n_blocks=5,
    n_heads=8,
    d_b=None,
    d_f=None,
    d_ff=None,
    max_d_rate=None,
    causal=True,
    warmup_steps=40000,
    length=None,
    m_1=None,
    centre=None,
    scale=None,
    unit_type=None,
    loss_fnc="BinaryCrossentropy",
    outp_act="Sigmoid",
    max_len=2048,
    map_type="DBNormalCDF",
    map_params=[None, None],
)
def predict(test_x,test_x_len=1,test_x_base_names=['defaul']):
    a= deepxi.infer(
		test_x=test_x,
		test_x_len=test_x_len,
		test_x_base_names=test_x_base_names,
		test_epoch=200,
		model_path='../model/mhanet-1.1c',
		out_type='y',
		gain='mmse-lsa',
		out_path='../test/out')
    return a
