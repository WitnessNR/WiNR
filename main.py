import subprocess 
import numpy as np
from cnn_bounds_full_with_LP import run as run_cnn_full
from cnn_bounds_full_core_with_LP import run as run_cnn_full_core
# from tensorflow.contrib.keras.api.keras import backend as K
#from nosub_main import run_main

import time as timing
import datetime

ts = timing.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

#Prints to log file
def printlog(s):
    print(s, file=open("logs/log_main_"+timestr+".txt", "a"))

def run_cnn(file_name, n_samples, eps_0, norm, core=True, activation='sigmoid', cifar=False, fashion_mnist=False, gtsrb=False):
    if core:
        if norm == 'i':
            return run_cnn_full_core(file_name, n_samples, eps_0, 105, 1, activation, cifar, fashion_mnist, gtsrb)
        elif norm == '2':
            return run_cnn_full_core(file_name, n_samples, eps_0, 2, 2, activation, cifar, fashion_mnist, gtsrb)
        if norm == '1':
            return run_cnn_full_core(file_name, n_samples, eps_0, 1, 105, activation, cifar, fashion_mnist, gtsrb)
    else:
        if norm == 'i':
            return run_cnn_full(file_name, n_samples, eps_0, 105, 1, activation, cifar, fashion_mnist, gtsrb)
        elif norm == '2':
            return run_cnn_full(file_name, n_samples, eps_0, 2, 2, activation, cifar, fashion_mnist, gtsrb)
        if norm == '1':
            return run_cnn_full(file_name, n_samples, eps_0, 1, 105, activation, cifar, fashion_mnist, gtsrb)


def run_all_general(file_name, num_image, eps_0, core=True, cifar=False, fashion_mnist=False, gtsrb=False):
    if len(file_name.split('_')) == 5:
        nlayer = file_name.split('_')[-3][0]
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    #for norm in ['i', '2', '1']
    for norm in ['i']:
        #eps, total_images, DeepCert_robust_number, PGD_falsified_number, PGD_DeepCert_unknown_number, PGD_aver_time, DeepCert_aver_time, PGD_DeepCert_aver_time, WiNR_robust_number, WiNR_falsified_number, WiNR_unknown_number, WiNR_aver_time, WiNR_PGD_robust_number, PGD_WiNR_total_falsified_number, WiNR_PGD_unknown_number, PGD_WiNR_aver_time, WiNR_PGD_aver_time, PGD_WiNR_total_aver_time = run_cnn(file_name, num_image, eps_0, norm, core=core, activation = 'sigmoid', cifar= cifar, fashion_mnist=fashion_mnist, gtsrb=gtsrb) 
        eps, total_images, WiNR_robust_number, WiNR_falsified_number, WiNR_unknown_number, WiNR_aver_time, WiNR_PGD_robust_number, PGD_WiNR_total_falsified_number, WiNR_PGD_unknown_number, PGD_WiNR_aver_time, WiNR_PGD_aver_time, PGD_WiNR_total_aver_time = run_cnn(file_name, num_image, eps_0, norm, core=core, activation = 'sigmoid', cifar= cifar, fashion_mnist=fashion_mnist, gtsrb=gtsrb) 
        printlog("PGD+Deepcert, WiNR, PGD+WiNR")
        if filters:
            printlog("model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}".format(file_name,nlayer,num_image,norm,filters,kernel_size))
        else:
            printlog("model name = {0}, numimage = {1}, norm = {2}, targettype = random".format(file_name,num_image,norm))
        printlog("eps = {:.5f}".format(eps))
        printlog("total_images = {}".format(total_images))
        #printlog("PGD+DeepCert: robust number = {}, falsified number = {}, unknown number = {}, PGD average time = {:.3f}, DeepCert average time = {:.3f}, PGD+DeepCert average time = {:.3f}".format(DeepCert_robust_number, PGD_falsified_number, PGD_DeepCert_unknown_number, PGD_aver_time, DeepCert_aver_time, PGD_DeepCert_aver_time))
        printlog("WiNR:         robust number = {}, falsified number = {}, unknown number = {}, average time = {:.3f}".format(WiNR_robust_number, WiNR_falsified_number, WiNR_unknown_number, WiNR_aver_time))
        printlog("PGD+WiNR:     robust number = {}, falsified number = {}, unknown number = {}, PGD average time = {:.3f}, WiNR average time = {:.3f}, PGD+WiNR average time = {:.3f}".format(WiNR_PGD_robust_number, PGD_WiNR_total_falsified_number, WiNR_PGD_unknown_number, PGD_WiNR_aver_time, WiNR_PGD_aver_time, PGD_WiNR_total_aver_time))
        printlog("-----------------------------------")

if __name__ == '__main__':
    
    printlog("-----------------------------------")
    run_all_general('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.1, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.5, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 1.0, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 1.5, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 2.0, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 2.5, fashion_mnist=True)
    
    run_all_general('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.1, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.5, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 1.0, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 1.5, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 2.0, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 2.5, fashion_mnist=True)
    
    
    run_all_general('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.1, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.5, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 1.0, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 1.5, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 2.0, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 2.5, fashion_mnist=True)
    
    run_all_general('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.1, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.5, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 1.0, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 1.5, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 2.0, fashion_mnist=True)
    run_all_general('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', num_image = 100, eps_0 = 2.5, fashion_mnist=True)
    
    
    run_all_general('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', num_image = 100, eps_0 = 0.1, cifar=True, core=False)
    run_all_general('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', num_image = 100, eps_0 = 0.5, cifar=True, core=False)
    run_all_general('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', num_image = 100, eps_0 = 1.0, cifar=True, core=False)
    run_all_general('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', num_image = 100, eps_0 = 1.5, cifar=True, core=False)
    run_all_general('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', num_image = 100, eps_0 = 2.0, cifar=True, core=False)
    run_all_general('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', num_image = 100, eps_0 = 2.5, cifar=True, core=False)
    
    # GTSRB 数据集网络
    run_all_general('models/gtsrb_cnn_5layer_sigmoid_myself.h5', num_image = 100, eps_0 = 0.1, gtsrb=True)
    run_all_general('models/gtsrb_cnn_5layer_sigmoid_myself.h5', num_image = 100, eps_0 = 0.5, gtsrb=True)
    run_all_general('models/gtsrb_cnn_5layer_sigmoid_myself.h5', num_image = 100, eps_0 = 1.0, gtsrb=True)
    run_all_general('models/gtsrb_cnn_5layer_sigmoid_myself.h5', num_image = 100, eps_0 = 1.5, gtsrb=True)
    run_all_general('models/gtsrb_cnn_5layer_sigmoid_myself.h5', num_image = 100, eps_0 = 2.0, gtsrb=True)
    run_all_general('models/gtsrb_cnn_5layer_sigmoid_myself.h5', num_image = 100, eps_0 = 2.5, gtsrb=True)
    
    # 101,920个节点的网络
    # run_all_general('models/fashion_mnist_cnn_8layer_35_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.010, fashion_mnist=True)
    # run_all_general('models/fashion_mnist_cnn_8layer_35_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.100, fashion_mnist=True)
    # run_all_general('models/fashion_mnist_cnn_8layer_35_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.500, fashion_mnist=True)
    # run_all_general('models/fashion_mnist_cnn_8layer_35_3_sigmoid_myself.h5', num_image = 100, eps_0 = 1.500, fashion_mnist=True)
    # run_all_general('models/fashion_mnist_cnn_8layer_35_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.025, fashion_mnist=True)
    # run_all_general('models/fashion_mnist_cnn_8layer_35_3_sigmoid_myself.h5', num_image = 100, eps_0 = 0.030, fashion_mnist=True)
    
    
    # LB = []
    # time = []
    # tables = [1,2,3]
    
    # log_name = 'logs/results_test_log.txt'
    # f = open(log_name, 'w')
    
    
    # for table in tables:
        
    #     print("===================================================", file=f)
    #     print("=================WiNR Running Table {} ============".format(table), file=f)
    #     print("===================================================", file=f)
        
    #     total_models = []
    #     total_resutls = []
        
    #     # pure cnn
    #     if table == 1:
    #         total_models.append('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5')
    #         total_resutls.append(run_all_general('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', fashion_mnist=True))
    #         total_models.append('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5')
    #         total_resutls.append(run_all_general('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', fashion_mnist=True))
    #         total_models.append('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5')
    #         total_resutls.append(run_all_general('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', fashion_mnist=True))
    #         total_models.append('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5')
    #         total_resutls.append(run_all_general('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', fashion_mnist=True))
    #         total_models.append('models/gtsrb_cnn_5layer_sigmoid_myself.h5')
    #         total_resutls.append(run_all_general('models/gtsrb_cnn_5layer_sigmoid_myself.h5', gtsrb=True))
            
    #     # general cnn
    #     if table == 2:
    #         total_models.append('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5')
    #         total_resutls.append(run_all_general('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', cifar=True, core=False))
                        
    #     # resnet
    #     if table == 3:
    #         total_models.append('models/cifar10_resnet_2_sigmoid_myself')
    #         total_resutls.append(run_all_general('models/cifar10_resnet_2_sigmoid_myself', core=False, cifar=True))
            
        
    #     for j in range(len(total_models)):
    #         print(total_models[j], file=f)
    #         print('eps  robust_number  unrobust_number  has_adv_false_number avg_runtime', file=f)
    #         results = total_resutls[j]
    #         print(results)
    #         for i in range(len(results)):
    #             print(results[i][0], '\t', results[i][1], '\t\t', results[i][2], '\t\t', results[i][3], '\t\t', results[i][4], '\t\t', file=f)
    #         print()
    #         print()
    
    
    # for table in tables:
        
    #     print("========================================================", file=f)
    #     print("========Without Substitution Running Table {} ==========".format(table), file=f)
    #     print("========================================================", file=f)
        
        
    #     total_models = []
    #     total_resutls = []
        
    #     # pure cnn
    #     if table == 1:
    #         total_models.append('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5')
    #         total_resutls.append(run_main('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
    #         total_models.append('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5')
    #         total_resutls.append(run_main('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
    #         total_models.append('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5')
    #         total_resutls.append(run_main('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
    #         total_models.append('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5')
    #         total_resutls.append(run_main('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
    #         total_models.append('models/gtsrb_cnn_5layer_sigmoid_myself.h5')
    #         total_resutls.append(run_main('models/gtsrb_cnn_5layer_sigmoid_myself.h5', 'cnn', 'gtsrb'))
            
    #     # general cnn
    #     if table == 2:
    #         total_models.append('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5')
    #         total_resutls.append(run_main('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', 'cnn', 'cifar10'))
                        
        
    #     for j in range(len(total_models)):
    #         print(total_models[j], file=f)
    #         print('eps  robust_number  unrobust_number  has_adv_false_number total_number total_runtime', file=f)
    #         results = total_resutls[j]
    #         print(results, file=f)
    #         print()
    #         print()
    
    # f.close()
