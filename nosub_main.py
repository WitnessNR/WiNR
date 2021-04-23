# -*- coding: utf-8 -*-

import argparse
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
import numpy as np
import datetime
import tensorflow as tf
import verifier

from skimage import transform
from skimage import exposure
from skimage import io
from preprocess import *
import matplotlib.pyplot as plt
import math


def run_main(netname, net_type, dataset):
    
    netname, net_type, dataset, x_test, y_test = parse_argument(netname, net_type, dataset)
    print('*******************')
    print('x_test.shape', x_test.shape)
    print('y_test.shape', y_test.shape)
    print('*******************')
    nn = load_model(netname)

    eps = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    #eps = [0.3]
    eps_dic = {}
    test_image_num = 100
    limit_time = 600

    for ite in eps:
        print()
        print('################')
        epsilon = ite
        print('epsilon: ', epsilon)

        correctly_classified_images = 0
        verified_images = 0
        false_adversarial_images = 0
        has_adv_images = 0
        time_sum = 0

        solve_all_test_image_start = datetime.datetime.now()
        for i in range(test_image_num):
            img = x_test[i,:,:]
            img = img.astype('float32')
            if dataset in ['mnist', 'cifar10', 'fashion_mnist']:
                img /= 255.
            label = None
            print(net_type, dataset)
            
            label, img = predict_label(nn, net_type, dataset, x_test, img)
            print('label: ', label)
            print('img shape: ', img.shape)

            
            if label == y_test[i]:
                correctly_classified_images += 1
                
                ub_img = np.clip(img + epsilon, 0, 1)
                lb_img = np.clip(img - epsilon, 0, 1)
                
                robust_flag, adv_image, adv_label, false_positive = verifier.verify_network_with_solver(ub_img, lb_img, label, nn, net_type, dataset)
                #print('adv_img.shape: ', adv_image.shape)

                if robust_flag:
                    verified_images += 1
                    print("img: ", i, "Verified. label: ", label)
                else:
                    if false_positive:
                        false_adversarial_images += 1
                        print("adversarial image predict label: ", adv_label)
                        print("solver solution Error! img ", i, " Verify Failed")
                    else:
                        save_adv_image = np.clip(adv_image * 255, 0, 255)
                        fashion_mnist_labels_names = ['T-shirt or top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                        cifar10_labels_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                        if dataset == 'fashion_mnist':
                            plt.imshow(save_adv_image.squeeze(), cmap='gray')
                            adv_label_str = fashion_mnist_labels_names[adv_label]
                        elif dataset == 'cifar10':
                            save_adv_image = save_adv_image.astype(np.int32)
                            plt.imshow(save_adv_image)
                            adv_label_str = cifar10_labels_names[adv_label]
                        elif dataset == 'gtsrb':
                            save_adv_image = save_adv_image.astype(np.int32)
                            plt.imshow(save_adv_image)
                            adv_label_str = str(adv_label)
                        else:
                            plt.imshow(save_adv_image.squeeze(), cmap='gray')
                            adv_label_str = str(adv_label)
            
                        save_path = 'adv_examples/'+ dataset + '_'+str(epsilon)+'_adv_image_'+str(i)+'_adv_label_'+adv_label_str +'.png'
                        plt.savefig(save_path)
                        
                        original_image = np.clip(img.astype(np.float32)*255,0,255)
                        if dataset == 'fashion_mnist':
                            plt.imshow(original_image.squeeze(), cmap='gray')
                            predict_label_str = fashion_mnist_labels_names[label]
                        elif dataset == 'cifar10':
                            original_image = original_image.astype(np.int32)
                            plt.imshow(original_image)
                            predict_label_str = cifar10_labels_names[label]
                        elif dataset == 'gtsrb':
                            original_image = original_image.astype(np.int32)
                            plt.imshow(original_image)
                            predict_label_str = str(label)
                        else:
                            plt.imshow(original_image.squeeze(), cmap='gray')
                            predict_label_str = str(label)
                       
                        save_path = 'adv_examples/'+ dataset +'_'+str(epsilon)+'_original_image_'+str(i)+'_predict_label_'+predict_label_str+'.png'
                        plt.savefig(save_path)
                        
                        has_adv_images += 1
                        print("adversarial image: ", adv_image, "adversarial label: ", adv_label, "correct label: ", label, "epsilon: ", epsilon)    

            else:
                print("img ", i, " not considered, correct_label: ", y_test[i], "classified label: ", label)

            time_sum = (datetime.datetime.now()- solve_all_test_image_start).seconds
            if time_sum >= limit_time:
                print('time_sum:', time_sum)
                print('limit_time:', limit_time)
                break
        solve_all_test_image_end = datetime.datetime.now()
        spend_time = (solve_all_test_image_end - solve_all_test_image_start).seconds
        print("All test images verify time: ", spend_time, "seconds")
        print('analysis precision ', verified_images, '/', correctly_classified_images)
        eps_dic[ite] = (ite, verified_images, has_adv_images, false_adversarial_images, correctly_classified_images, spend_time)

    print('*****************')
    for ite in eps:
        print(ite)
        print(eps_dic[ite])
    return eps_dic

# if __name__ == '__main__':
#     tables = [2]
    
#     log_name = 'logs/total_results_v1_log.txt'
#     f = open(log_name, 'a')
    
#     for table in tables:
        
#         print("==================================================", file=f)
#         print("================ Running Table {} ================".format(table), file=f)
#         print("==================================================", file=f)
        
#         total_models = []
#         total_resutls = []
        
#         # pure cnn
#         if table == 1:
#             total_models.append('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5')
#             total_resutls.append(run_main('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
#             total_models.append('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5')
#             total_resutls.append(run_main('models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
#             total_models.append('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5')
#             total_resutls.append(run_main('models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
#             total_models.append('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5')
#             total_resutls.append(run_main('models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself.h5', 'cnn', 'fashion_mnist'))
#             total_models.append('models/gtsrb_cnn_5layer_sigmoid_myself.h5')
#             total_resutls.append(run_main('models/gtsrb_cnn_5layer_sigmoid_myself.h5', 'cnn', 'gtsrb'))
            
#         # general cnn
#         if table == 2:
#             total_models.append('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5')
#             total_resutls.append(run_main('models/cifar10_cnn_lenet_averpool_sigmoid_myself.h5', 'cnn', 'cifar10'))
                        
        
#         for j in range(len(total_models)):
#             print(total_models[j], file=f)
#             print('eps  robust_number  unrobust_number  has_adv_false_number total_number avg_runtime', file=f)
#             results = total_resutls[j]
#             print(results, file=f)
#             # for i in range(len(results)):
#             #     print(results[i][0], '\t', results[i][1], '\t\t', results[i][2], '\t\t', results[i][3], '\t\t', results[i][4], '\t\t', results[i][5], file=f)
#             print()
#             print()
    
#     f.close()
