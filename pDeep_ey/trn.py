import model.fragmentation_config as fconfig
import model.lstm_tf as lstm
from model.load_data import load_folder_as_buckets as load_folder
from model.bucket_utils import merge_buckets, print_buckets, count_buckets
import numpy as np
from absl import logging
import tensorflow as tf
import os

ion_types = ['internal{}']
mod_config = fconfig.HCD_CommonMod_Config()         #check modification
mod_config.SetFixMod(['Carbamidomethyl[C]'])
#mod_config.varmod = ["Oxidation[M]"]
mod_config.SetIonTypes(ion_types)
mod_config.time_step = 100
mod_config.min_var_mod_num = 0
mod_config.max_var_mod_num = 15

pdeep = lstm.IonLSTM(mod_config)

pdeep.learning_rate = 0.001
pdeep.layer_size = 256
pdeep.batch_size = 256
pdeep.BuildModel(input_size = 98, output_size = mod_config.GetTFOutputSize(), nlayers = 2)      #GetTFOutputSize: len(ion_types)*max_ion_charge ex. b1(charge1), b2(charge1)..
                                                                                                #input_size 왜 98?
pdeep.epochs = 100
n = 100000000

out_folder = './tf-models/0928/model1/'
model_name = '0928model1.ckpt' # the model is saved as ckpt file #0.1

try:
    os.makedirs(out_folder)
except:
    pass
    
    
buckets = {}
PT_NCE30 = "./data/data1" # folder containing plabel files
buckets = merge_buckets(buckets, load_folder(PT_NCE30, mod_config, nce = 0.30, instrument = 'Lumos', max_n_samples = n))
# you can add more plabel-containing folders here

print('[I] train data:')
print_buckets(buckets, print_peplen = False)
buckets_count = count_buckets(buckets)
print(buckets_count)
print(buckets_count["total"])


pdeep.TrainModel(buckets, save_as = os.path.join(out_folder, model_name))

pdeep.close_session()
