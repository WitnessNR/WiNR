import subprocess 
import numpy as np
from cnn_bounds_full_with_LP import run as run_cnn_full
from cnn_bounds_full_core_with_LP import run as run_cnn_full_core
from tensorflow.contrib.keras.api.keras import backend as K
from WithoutSub.nosub_main import run_main

import time as timing
import datetime

def run_cnn(file_name, n_samples, norm, core=True, activation='sigmoid', cifar=False, fashion_mnist=False, gtsrb=False):
    if core:
        if norm == 'i':
            return run_cnn_full_core(file_name, n_samples, 105, 1, activation, cifar, fashion_mnist, gtsrb)
        elif norm == '2':
            return run_cnn_full_core(file_name, n_samples, 2, 2, activation, cifar, fashion_mnist, gtsrb)
        if norm == '1':
            return run_cnn_full_core(file_name, n_samples, 1, 105, activation, cifar, fashion_mnist, gtsrb)
    else:
        if norm == 'i':
            return run_cnn_full(file_name, n_samples, 105, 1, activation, cifar, fashion_mnist, gtsrb)
        elif norm == '2':
            return run_cnn_full(file_name, n_samples, 2, 2, activation, cifar, fashion_mnist, gtsrb)
        if norm == '1':
            return run_cnn_full(file_name, n_samples, 1, 105, activation, cifar, fashion_mnist, gtsrb)


def run_all_general(file_name, num_image = 100, core=True, cifar=False, fashion_mnist=False, gtsrb=False):
    if len(file_name.split('_')) == 5:
        nlayer = file_name.split('_')[-3][0]
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    #for norm in ['i', '2', '1']
    for norm in ['i']:
        return run_cnn(file_name, num_image, norm, core=core, activation = 'sigmoid', cifar= cifar, fashion_mnist=fashion_mnist, gtsrb=gtsrb) 


if __name__ == '__main__':
    LB = []
    time = []
    tables = [1,2,3]
    
    log_name = 'logs/results_test_log.txt'
    f = open(log_name, 'w')
    
    
    for table in tables:
        
        print("===================================================", file=f)
        print("=================WiNR Running Table {} ============".format(table), file=f)
        print("===================================================", file=f)
        
        total_models = []
        total_resutls = []
        
        # pure cnn
        if table == 1:
            total_models.append('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5')
            total_resutls.append(run_all_general('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', fashion_mnist=True))
            total_models.append('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5')
            total_resutls.append(run_all_general('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', fashion_mnist=True))
            total_models.append('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5')
            total_resutls.append(run_all_general('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', fashion_mnist=True))
            total_models.append('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5')
            total_resutls.append(run_all_general('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', fashion_mnist=True))
            total_models.append('models/gtsrb_cnn_5layer_sigmoid_myself.h5')
            total_resutls.append(run_all_general('models/gtsrb_cnn_5layer_sigmoid_myself.h5', gtsrb=True))
            
        # general cnn
        if table == 2:
            total_models.append('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5')
            total_resutls.append(run_all_general('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', cifar=True, core=False))
                        
        # resnet
        if table == 3:
            total_models.append('models/cifar10_resnet_2_sigmoid_myself')
            total_resutls.append(run_all_general('models/cifar10_resnet_2_sigmoid_myself', core=False, cifar=True))
        
        for j in range(len(total_models)):
            print(total_models[j], file=f)
            print('eps  robust_number  unrobust_number  has_adv_false_number avg_runtime', file=f)
            results = total_resutls[j]
            print(results)
            for i in range(len(results)):
                print(results[i][0], '\t', results[i][1], '\t\t', results[i][2], '\t\t', results[i][3], '\t\t', results[i][4], '\t\t', file=f)
            print()
            print()
    
    for table in tables:
        
        print("========================================================", file=f)
        print("========Without Substitution Running Table {} ==========".format(table), file=f)
        print("========================================================", file=f)
        
        total_models = []
        total_resutls = []
        
        # pure cnn
        if table == 1:
            total_models.append('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5')
            total_resutls.append(run_main('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
            total_models.append('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5')
            total_resutls.append(run_main('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
            total_models.append('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5')
            total_resutls.append(run_main('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
            total_models.append('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5')
            total_resutls.append(run_main('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
            total_models.append('models/gtsrb_cnn_5layer_sigmoid_myself.h5')
            total_resutls.append(run_main('models/gtsrb_cnn_5layer_sigmoid_myself.h5', 'cnn', 'gtsrb'))
            
        # general cnn
        if table == 2:
            total_models.append('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5')
            total_resutls.append(run_main('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', 'cnn', 'cifar10'))
                        
        
        for j in range(len(total_models)):
            print(total_models[j], file=f)
            print('eps  robust_number  unrobust_number  has_adv_false_number total_number total_runtime', file=f)
            results = total_resutls[j]
            print(results, file=f)
            print()
            print()
    
    f.close()
