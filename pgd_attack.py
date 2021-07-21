# from https://github.com/skmda37/Adversarial_Machine_Learning_Tensorflow

import tensorflow as tf
# from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time

class AdversarialAttack:
    def __init__(self, model, eps):
        """
        :param model: instance of tf.keras.Model that is used to generate adversarial examples with attack
        :param eps: float number - maximum perturbation size of adversarial attack
        """
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()# Loss that is used for adversarial attack
        self.model = model     # Model that is used for generating the adversarial examples
        self.eps = eps         # Threat radius of adversarial attack
        self.specifics = None  # String that contains all hyperparameters of attack
        self.name = None       # Name of the attack - e.g. PGD
        
class PgdRandomRestart(AdversarialAttack):
    def __init__(self, model, eps, alpha, num_iter, restarts):
        """
        :param model: instance of tf.keras.Model that is used to generate adversarial example
        :param eps: float number - maximum perturbation size for adversarial attack
        :param alpha: float number - step size in adversarial attack
        :param num_iter: integer - number of iterations of pgd during one restart iterarion
        :param restarts: integer - number of restarts
        """
        super().__init__(model, eps)
        self.name = "PGD With Random Restarts"
        self.specifics = "PGD With Random Restarts - "\
                         f"eps: {eps} - alpha: {alpha} - "\
                         f"num_iter: {num_iter} - restarts:{restarts}"
        self.alpha = alpha
        self.num_iter = num_iter
        self.restarts = restarts
        self.model = model
        
    def __call__(self, clean_images, true_labels, time_limit, predict_label, false_positive=False):
        """
        :param clean_image: tf.Tensor - shape (n,h,w,c) - clean image will be transformed into adversarial examples
        :param true_labels: tf.Tensor - shape (n,) - true lables of clean_images
        :return: adversarial examples generated with PGD with random restarts
        """
        print("time_limit: ", time_limit)
        attack_start_time = time.time()
        # Get loss on clean_images
        # max_loss = tf.keras.losses.sparse_categorical_crossentropy(true_labels, self.model(clean_images)) version - 1.12
        max_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)(true_labels, self.model(clean_images))
        # max_X contains adversarial examples and is updated after each reastart
        max_X = clean_images[:,:,:,:]
        
        # Start restart loop
        for i in range(self.restarts):
            # Get random perturbation uniformly in l infinity epsilon ball
            random_delta = 2 * self.eps * tf.random.uniform(shape=clean_images.shape) - self.eps
            # Add random perturbation
            X = clean_images + random_delta
            
            break_flag = False
            
            # Start projective gradient descent from X
            for j in range(self.num_iter):
                # Track gradients
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    # Only gradients w.r.t. X are taken NOT model parameters
                    tape.watch(X)
                    pred = self.model(X)
                    loss = self.loss_obj(true_labels, pred)
                    
                # Get gradients of loss w.r.t. X
                gradients = tape.gradient(loss, X)
                # Compute perturbation as step size times sign of gradients
                perturbation = self.alpha * tf.sign(gradients)
                # Update X by adding perturbation
                X = X + perturbation
                # Make sure X did not leave l infinity epsilon ball around clean_images
                X = tf.clip_by_value(X, clean_images - self.eps, clean_images + self.eps)
                # Make sure X has entries between 0 and 1
                X = tf.clip_by_value(X, 0, 1)
                
                attack_end_time = (time.time() - attack_start_time)
                if attack_end_time >= time_limit:
                    print("attack end time: ", attack_end_time)
                    break_flag = True
                    break
            
            if break_flag:
                break
            
            # Get crossentroby loss for each_image in X
            # loss_vector = tf.keras.losses.sparse_categorical_crossentropy(true_labels, self.model(X)) version - 1.12
            loss_vector = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE)(true_labels, self.model(X))
            
            # Mask is 1D tensor where true values are the rows of images that have higher loss than previous restarts
            mask = tf.greater(loss_vector, max_loss)
            # Update max_loss
            max_loss = tf.where(mask, loss_vector, max_loss)
            """ 
            We cannot do max_X[mask] = X[mask] like in numpy. We need mask that fits shape of max_X.
            Keep in mind that we want to select the rows that are True in the 1D tensor mask.
            We can simply stack the mask along the dimensions of max_X to select each desired row later.
            """
            # Create 2D mask of shape (max_X.shape[0], max_X.shape[1])
            multi_mask = tf.stack(max_X.shape[1] * [mask], axis=-1)
            # Create 3D mask of shape (max_X.shape[0], max_X.shape[1], max_X.shape[2])
            multi_mask = tf.stack(max_X.shape[2] * [multi_mask], axis=-1)
            # Create 4D mask of shape (max_X.shape[0], max_X.shape[1], max_X.shape[2], max_X.shape[3])
            multi_mask = tf.stack(max_X.shape[3] * [multi_mask], axis=-1)
            
            # Replace adversaial examples max_X[i] that have smaller loss than X_[i] with X[i]
            max_X = tf.where(multi_mask, X, max_X)
            
            adv_example_label = self.model.predict(max_X)
            adv_example_label = np.argmax(adv_example_label)
            if adv_example_label != predict_label:
                return max_X
        
        # return adversarial exmaples
        return max_X

def attack_visual_demo(model, Attack, images, labels, attack_kwargs):
    """ Demo of  adversarial attack on 20 images
    :param model: tf.keras.Model
    :param Attack: type attacks.AdversarialAttack
    :param attack_kwargsL dictionary - keyword arguments to call of instance of Attack
    :param images: tf.Tensor - shape (20, h, w, c)
    :param labels: tf.Tensor - shape (20,)
    :return Nothing
    """
    print("enter attack_visual_demo")
    #assert images.shape[0] == 20
    
    attack = Attack(model=model, **attack_kwargs)
    
    # fig, axs = plt.subplots(4,11,figsize=(15,8))
    
    # # Plot model preditions on clean images
    # for i in range(4):
    #     for j in range(5):
    #         image = images[5 * i + j]
    #         label = labels[5 * i + j]
    #         ax = axs[i, j]
    #         # show_image = tf.Session().run(image)
    #         # show_image = np.clip(show_image * 255, 0, 255)
    #         # show_image = np.squeeze(show_image)
    #         # ax.imshow(show_image, cmap="gray")
    #         ax.imshow(tf.squeeze(image), cmap="gray")
    #         ax.axis("off")
            
    #         # prediction = model.predict(np.expand_dims(tf.Session().run(image), axis=0))
    #         # prediction = np.argmax(prediction, axis=1)
    #         # label = tf.Session().run(label)
            
    #         prediction = model(tf.expand_dims(image, axis=0))
    #         prediction = tf.math.argmax(prediction, axis=1)
    #         prediction = tf.squeeze(prediction)
    #         color = "green" if prediction.numpy() == label.numpy() else "red"
            
    #         ax.set_title("Pred: "+str(prediction.numpy()), color=color, fontsize=18)
            
    print("##################")
    # # plot empty column
    # for i in range(4):
    #     axs[i,5].axis("off")
    
    # Set attack inputs
    attack_inputs = (images, labels)
    
    # Get adversarial examples
    adv_examples = attack(*attack_inputs)
    print("len(adv_examples): ", adv_examples.shape)
    prediction = model(adv_examples)
    prediction = tf.math.argmax(prediction, axis=1)
    print("prediction: ", prediction)
    # # Plot model predictions on adversarial examples
    # for i in range(4):
    #     for j in range(5):
    #         image = adv_examples[5 * i + j]
    #         label = labels[5 * i + j]
    #         ax = axs[i, 6 + j]
    #         # show_image = tf.Session().run(image)
    #         # show_image = np.clip(show_image * 255, 0, 255)
    #         # show_image = np.squeeze(show_image)
    #         # ax.imshow(show_image, cmap="gray")
    #         ax.imshow(tf.squeeze(image), cmap="gray")
    #         ax.axis("off")
            
    #         prediction = model(tf.expand_dims(image, axis=0))
    #         prediction = tf.math.argmax(prediction, axis=1)
    #         prediction = tf.squeeze(prediction)
    #         # print(prediction.numpy())
    #         # print(label.numpy())
    #         color = "green" if prediction.numpy() == label.numpy() else "red"
    #         # prediction = model.predict(np.expand_dims(tf.Session().run(image), axis=0))
    #         # prediction = np.argmax(prediction, axis=1)
    #         # label = tf.Session().run(label)
    #         # color = "green" if prediction == label else "red"
            
    #         ax.set_title("Pred: "+str(prediction.numpy()), color=color, fontsize=18)
    
    print("-----------------show images----------------")
    # plot text
    # plt.subplots_adjust(hspace=0.4)
    # plt.figtext(0.16, 0.93, "Model Prediction on Clean Images", fontsize=18)
    # plt.figtext(0.55, 0.93, "Model Prediction on Adversarial Examples", fontsize=18)
    # plt.figtext(0.1, 1, "Adversarial Attack: "+attack.specifics, fontsize=24)
    # plt.show()


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    x_test_print = (x_test.reshape(10000,28,28,1).astype("float32")) / 255.
    print("1: ", x_test_print.shape)
    print("2: ", type(x_test_print))
    
    y_test_print = y_test.astype("float32")
    print("3: ", y_test_print.shape)
    print("4: ", type(y_test_print))

    x_train = tf.constant(x_train.reshape(60000,28,28,1).astype("float32") / 255)
    x_test = tf.constant(x_test.reshape(10000,28,28,1).astype("float32") / 255)

    y_train = tf.constant(y_train.astype("float32"))
    y_test = tf.constant(y_test.astype("float32"))
    
    
    
    # x_train = (x_train.reshape(60000,28,28,1).astype("float32")) / 255.
    # x_test_print = (x_test.reshape(10000,28,28,1).astype("float32")) / 255.

    # y_train = y_train.astype("float32")
    # y_test_print = y_test.astype("float32")

    my_model = load_model('models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5')

    Attack = PgdRandomRestart
    attack_kwargs = {"eps": 0.005, "alpha": 0.25/40, "num_iter": 40, "restarts": 10}
    
    attack_visual_demo(my_model, Attack, x_test[:1], y_test[:1], attack_kwargs)
    print("finished!")