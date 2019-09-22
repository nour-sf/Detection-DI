#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:00:12 2019

@author: nour
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:22:39 2019

@author: nour
"""

import subprocess
import numpy as np

#for epochs in np.arange(10,60,10):
for lr_pow in np.arange(-5.0,-1.0,1.0):
    for droprate in np.arange(0.1,0.6,0.1):
        for num_layers in np.arange(2,5,1):
            lr = 10**(lr_pow)
            epochs = 40
            batch = 8
            num_features = 8
            SNR = 0.75
            decay = lr/batch
            process0 = subprocess.Popen([
                    "python CNN_keras_nour.py --datapath ./../../../../data2/visitor2/Direct-Imaging-Project/DI_dataset/CNN_training/master_training.npy --testpath ./../../../../data2/visitor2/Direct-Imaging-Project/DI_dataset/CNN_training/master_test.npy --psfpath ./../../../../data2/visitor2/Direct-Imaging-Project/DI_dataset/CNN_training/tinyPSF.npy --epoch {} --batch_size {} --num_features {} --lr {} --decay {} --droprate {} --num_layers {} --cv_num 5 --SNR {} --checkpt ./output2/double_conv/epo{}_{}bat_{}feat_{}lr_{}decay_{}drop_{}nlayers ".format(
                    epochs, batch, num_features, lr, decay, droprate, num_layers,SNR, epochs, batch, num_features, lr, decay, droprate, num_layers)
                    ], shell=True)
            process0.wait()