import argparse
import os
import sys
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input,Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.models import Model,Sequential
from keras.models import load_model
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

# added by Nour for most of the folliwing imports
from sklearn.model_selection import train_test_split
from IPython.display import Image, display, SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# from hyperas.distributions import uniform

# K.set_image_dim_ordering('tf')
print(K.image_data_format())

## required for efficient GPU use
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)



class CNN():
    def __init__(self, datapath,psfpath,testpath, batch_size=32, epochs=100,
                 droprate=0.5, img_row=64, img_col=64,
                 num_features=32, num_class=2,
                 lr=0.0001, c_dim=1,
                 checkpoint_dir='checkpoint', CV_num=1, dense_unit=128, model_type='customised_CNN',SNR=10,
                 valid_size=0.2,decay = 0.0,num_layers=4):
        self.data = np.load(datapath)
        self.psf_pl = np.load(psfpath)
        self.test_data = np.load(testpath)
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.droprate = droprate
        self.img_rows = img_row
        self.img_cols = img_col
        self.num_layers = num_layers
        self.lr = lr
        self.num_class = num_class
        self.num_features = num_features
        self.c_dim = c_dim
        self.CV_num = CV_num
        self.dense_unit = dense_unit
        self.model_type = model_type
        self.valid_size = valid_size
        self.decay = decay
        self.SNR = SNR
        print('data shape: {}'.format(self.data.shape))

        self.run_model()
        
    # def data_preprocess_new(self,data, c_ratio,psf_library,repeats):
    #     '''
    #     Function to create postive exmaples from negative examples
    #     :param data: Negative exmaples in the form of (no.,img_w,img_h, channel)
    #     :param c_ratio: contrast ratio between max(speckle) and max(planet)
    #     :return:  postive examples with random numbers of injected planet and negative examples, normalised into range of [0,1]
    #     '''
    #     ## initialise array
    #     injected_samples = np.zeros([len(data)*(repeats*3), 64, 64])
    #     #pos_label = np.zeros([len(data)*(repeats*3), 64, 64])
    #     #neg_label = np.zeros([len(data), 64, 64])
    #
    #     ##inject planets
    #     for i in range(len(data)):
    #         new_img= self.inject_planet_batch(data[i].reshape(64, 64), psf_library = psf_library, c_ratio=c_ratio,x_bound=[4,61],y_bound=[4,61],no_blend=True,repeats = repeats)
    #         injected_samples[i*(repeats*3):i*(repeats*3)+(repeats*3)] += new_img
    #         #pos_label[i*(repeats*3):i*(repeats*3)+(repeats*3), :, :] += pos_label_batch
    #     '''
    #     Transform both pos and neg examples into [0,1]
    #     This step is optional , the initial rationale is that, if we train our CNN in terms of [0,1]
    #     we can always reduce real data to [0,1] and feed it into the CNN. And CNN may work better with [0,1] for every image.
    #     '''
    #     # normalised_injected = self.local_normal(injected_samples)
    #     # nor_data = self.local_normal(data)
    #
    #     # combine positive and negative examples into one stack.
    #     dataset = np.zeros([int(len(data) * (repeats*3+1)), 64, 64])
    #     dataset[:len(data)*(repeats*3)] += injected_samples
    #     dataset[len(data)*(repeats*3):] += data
    #
    #     ##label
    #     pos_label = np.ones(len(dataset[:len(data)*(repeats*3)]))
    #     neg_label = np.zeros(len(data))
    #     # combine positive and negative labels into one stack.
    #     all_label = np.append(pos_label, neg_label)
    #     print("label", all_label.shape)
    #     return dataset.reshape(-1, 64, 64, 1), all_label
    
    # def inject_planet_batch(self,data, psf_library, c_ratio=[0.1,0.2], x_bound=[4, 61], y_bound=[4, 61], no_blend=False,repeats=10):
    #     """Inject planet into random location within a frame
    #     data: single image
    #     psf_library: collection of libarary (7x7)
    #     c_ratio: the contrast ratio between max(speckle) and max(psf)*, currently accepting a range
    #     x_bound: boundary of x position of the injected psf, must be within [0,64-7]
    #     y_bound: boundary of y position of the injected psf, must be within [0,64-7]
    #     no_blend: optional flag, used to control whether two psfs can blend into each other or not, default option allows blending.
    #
    #     """
    #     def dist(dist1, *args):
    #         dist_list = []
    #         for centre in args:
    #             x,y = np.array(dist1),np.array(centre)
    #             distance = np.sqrt(np.sum((x-y)**2))
    #             dist_list.append(distance)
    #         return np.array(dist_list)
    #     image_stack = []
    #     pos_label_stack = []
    #     for repeat in range(repeats):
    #         pl_num = 1
    #         while pl_num <4:
    #             image = data.copy()
    #             pos_label = np.zeros([64, 64])
    #             used_xy = np.array([])
    #             c_prior = np.linspace(c_ratio[0], c_ratio[1], 100)
    #             if x_bound[0] < 4 or x_bound[0] > 61:
    #                 raise Exception("current method only injects whole psf")
    #             if y_bound[0] < 4 or y_bound[0] > 61:
    #                 raise Exception("current method only injects whole psf")
    #
    #             for num in range(pl_num):
    #                 while True:
    #                     np.random.shuffle(c_prior)
    #                     psf_idx = np.random.randint(0, high=psf_library.shape[0])
    #                     Nx = np.random.randint(x_bound[0], high=x_bound[1])
    #                     Ny = np.random.randint(y_bound[0], high=y_bound[1])
    #                     if len(used_xy) == 0:
    #                         pass
    #                     else:
    #                         if no_blend:
    #                             if np.any(dist([Nx, Ny], used_xy) < 3):
    #                                 pass
    #                         else:
    #                             if np.any(np.array([Nx, Ny]) == used_xy):
    #                                 pass
    #                     if dist([Nx, Ny], (32.5, 32.5)) < 4:
    #                         pass
    #                     else:
    #                         planet_psf = psf_library[psf_idx]
    #                         brightness_f = c_prior[0] * np.max(image) / np.max(planet_psf)
    #                         image[Ny - 4:Ny + 3, Nx - 4:Nx + 3] += planet_psf * brightness_f
    #                         used_xy = np.append(used_xy, [Nx, Ny]).reshape(-1, 2)
    #                         pos_label[Ny - 4:Ny + 3, Nx - 4:Nx + 3] = 1
    #                         break
    #             image_stack.append(image)
    #             pos_label_stack.append(pos_label)
    #             pl_num+=1
    #
    #     return np.array(image_stack).reshape(-1,64,64)

    def data_preprocess(self, data, SNR):
        """Data preprocess stage. Here each negative example is duplicated to produce a postive example by injecting a planet psf onto it.
        The injection method is a separate method and can be changed at any time.
        Labels for both training and test data is also created here. """
        ## inject planet for train_data
        injected_samples = np.zeros([len(data), self.img_rows, self.img_cols])

        for i in range(len(data)):
            new_img, Nx, Ny = self.SNR_injection(data[i].reshape(self.img_rows, self.img_cols), self.psf_pl, SNR=SNR)
            injected_samples[i] += new_img
        normalised_injected = self.local_normal(injected_samples)
        nor_data = self.local_normal(data)

        dataset = np.zeros([int(len(data) * 2), self.img_rows, self.img_cols])
        dataset[:len(data)] += normalised_injected
        dataset[len(data):] += nor_data

        label = np.zeros((len(dataset)))
        label[:len(data)] += 1
        print("label size =", label.shape)
        print("train data size=", dataset.shape)
        print("label sum=", np.sum(label))

        return dataset.reshape(-1, self.img_rows, self.img_cols, self.c_dim), label
    def SNR_injection(self, data, tinyPSF, SNR=20, verbose=False, num_pixel=4):
        """Planet injection method. """
        pl_PSF = tinyPSF
        pad_length = int(pl_PSF.shape[0] / 2)
        pad_data = np.pad(data, ((pad_length, pad_length), (pad_length, pad_length)), 'constant',
                          constant_values=(100000))
        width = int(num_pixel / 2)
        while True:

            Nx = np.random.randint(0, high=self.img_rows)
            Ny = np.random.randint(0, high=self.img_rows)
            aperture = pad_data[Ny + 19 - width:Ny + 19 + width, Nx + 19 - width:Nx + 19 + width]
            aperture = aperture[aperture < 100000]
            noise_std = np.std(aperture)
            FWHM_contri = np.sum(pl_PSF[19 - width:19 + width, 19 - width:19 + width])
            pl_brightness = (noise_std * SNR * len(aperture.flatten()) / FWHM_contri)

            if np.max(data) > np.max(pl_PSF * pl_brightness):
                break
            else:
                pass

        pad_data[Ny:Ny + pad_length * 2, Nx:Nx + pad_length * 2] += pl_PSF * pl_brightness
        if verbose:
            print("planet_PSF_signal=", np.sum(pl_PSF * pl_brightness))
            print("planet_PSF_FWHMsignal=",
                  np.sum(pl_PSF[19 - width:19 + width, 19 - width:19 + width] * pl_brightness))
            print("Peak planet signal=", np.max(pl_PSF[19 - width:19 + width, 19 - width:19 + width] * pl_brightness))
            print("Peak speckle signal=", np.max(data))
            print("noise std=", noise_std)
            plt.imshow(pad_data[pad_length:pad_length + self.img_rows, pad_length:pad_length + self.img_rows])
        return pad_data[pad_length:pad_length + self.img_rows, pad_length:pad_length + self.img_rows], Nx, Ny

    def local_normal(self, data):
        """Perform image by image normalisation. Maximum and Minimum value of each image is extracted and used to create an normalised image between [0,1]"""
        new_imgs_list = []
        for imgs in data:
            local_min = np.min(imgs)
            new_imgs = (imgs - local_min) / np.max(imgs - local_min)
            new_imgs_list.append(new_imgs)
        return np.array(new_imgs_list).reshape(-1, self.img_rows, self.img_cols)

    def return_heatmap(self, model, org_img,normalise = True):
        """CAM implementation here. An activation heatmap is produced for every test images. """
        test_img = model.output[:, 1]
        if self.model_type == 'simple':
            last_conv_layer = model.get_layer('conv2d_3')
        else:
            last_conv_layer = model.get_layer('conv2d_6')
        grads = K.gradients(test_img, last_conv_layer.output)[0]

        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        message = K.print_tensor(pooled_grads, message='pool_grad = ')
        iterate = K.function([model.input],
                             [message, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([org_img.reshape(-1, self.img_rows, self.img_cols, self.c_dim)])
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        if normalise:
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
        return heatmap

    def plot_heatmap(self, heatmap,diff, index, cv_num):
        """Plotting function to show the heatmap and planet location of the corresponding test image. """
        fig = plt.figure(figsize=(16, 8))

        grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                         nrows_ncols=(1, 2),
                         axes_pad=0.15,
                         share_all=True,
                         )

        # Add data to image grid
        im = grid[0].imshow(heatmap)
        im = grid[1].imshow(diff)

        plt.savefig(os.path.join(self.checkpoint_dir, 'heatmap_{}_{}'.format(index, cv_num)),bbox_inches='tight')

    def model_fn(self, model_type):
        """Architetures for different models are presented here """
        if not os.path.exists(os.path.join(self.checkpoint_dir, 'ckt')):
            os.makedirs(os.path.join(self.checkpoint_dir, 'ckt'))

        filter_pixel = 3

        # input image dimensions
        input_shape = (self.img_rows, self.img_cols, self.c_dim)
        def conv_block(input, filter_pixel=3, num_features=32, activation='relu', stride=1, double_conv=False):
            if double_conv:
                conv_layer = Conv2D(num_features, kernel_size=(filter_pixel, filter_pixel), padding="same",
                                    activation=activation, strides=stride,
                                    data_format="channels_last")(input)
                BN_layer = BatchNormalization(axis=3)(conv_layer)
                conv_layer2 = Conv2D(num_features, kernel_size=(filter_pixel, filter_pixel), padding="same",
                                     activation=activation, strides=stride,
                                     data_format="channels_last")(BN_layer)
                final_BN_layer = BatchNormalization(axis=3)(conv_layer2)
            else:
                conv_layer = Conv2D(num_features, kernel_size=(filter_pixel, filter_pixel), padding="same",
                                    activation=activation, strides=stride,
                                    data_format="channels_last")(input)
                final_BN_layer = BatchNormalization(axis=3)(conv_layer)
            return final_BN_layer

        # Start Neural Network
        self.model = Sequential()

        if model_type == 'vgg':
            # convolution 1st layer
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu',
                       input_shape=input_shape, dim_ordering="tf"))  # 0
            self.model.add(BatchNormalization(axis=3))
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu',
                       input_shape=(64, 64, 1), dim_ordering="tf"))  # 0
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))

            # convolution 2nd layer
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))
            #         #
            # convolution 3rd layer
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))
            # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  padding="same", dim_ordering="tf"))
            # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", padding="valid"))
            # model.add(LeakyReLU())

            # Fully connected 1st layer
            self.model.add(Flatten())  # 7
            self.model.add(Dense(self.dense_unit, use_bias=False, activation='relu'))  # 13
            # self.model.add(LeakyReLU()) #14
            self.model.add(Dropout(self.droprate))  # 15

            # Fully connected final layer
            self.model.add(Dense(2))  # 8
            self.model.add(Activation('sigmoid'))  # 9
            self.model.compile(loss=keras.losses.binary_crossentropy,
                               optimizer=keras.optimizers.Adam(lr=self.lr,decay=self.decay),
                               metrics=['accuracy'])

        elif model_type == 'simple':
            # convolution 1st layer
            self.model.add(
                Conv2D(self.num_features, kernel_size=(filter_pixel, filter_pixel), padding="same", activation='relu',
                       input_shape=input_shape, dim_ordering="tf", kernel_initializer='random_uniform'))  # 0
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

            # convolution 2nd layer
            self.model.add(Conv2D(self.num_features * 2, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  border_mode="same", dim_ordering="tf", kernel_initializer='random_uniform'))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
            self.model.add(Conv2D(self.num_features * 4, kernel_size=(filter_pixel, filter_pixel), activation='relu',
                                  border_mode="same", dim_ordering="tf", kernel_initializer='random_uniform'))  # 1
            self.model.add(BatchNormalization(axis=3))
            self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
            #
            # Fully connected 1st layer
            self.model.add(Flatten())  # 7
            self.model.add(Dense(self.dense_unit, use_bias=False, kernel_initializer='random_uniform', activation='relu'))  # 13
            # self.model.add(LeakyReLU()) #14
            self.model.add(Dropout(self.droprate))  # 15
            #self.model.add(Dropout({{uniform(0, 0.6)}})) 

            # Fully connected final layer
            self.model.add(Dense(2, use_bias=False, kernel_initializer='random_uniform', activation='sigmoid'))
            self.model.compile(loss=keras.losses.binary_crossentropy,
                               optimizer=keras.optimizers.Adam(lr=self.lr,decay=self.decay),
                               metrics=['accuracy'])

        # create a model with variable number of layers to be optimised
        elif model_type == 'customised_CNN':
            image = Input(shape=input_shape)
            cnn_layer = image
            for layer in range(self.num_layers):
                cnn_layer = conv_block(cnn_layer, filter_pixel=filter_pixel, num_features=self.num_features,double_conv=False)
                cnn_layer = MaxPooling2D(pool_size=(2, 2), data_format="channels_last", padding="valid")(cnn_layer)
            flatten_array = Flatten()(cnn_layer)
            dense_layer = Dense(self.dense_unit,activation='relu')(flatten_array)
            decision_layer = Dense(2,activation='sigmoid')(dense_layer)
            self.model = Model(inputs=image, outputs=decision_layer)
            self.model.compile(loss='binary_crossentropy',
                               optimizer=keras.optimizers.Adam(lr=self.lr, decay=self.decay),
                               metrics=['accuracy'])
            self.model.summary()

#___________________________________________________________________________________________________
#___________________________________________________________________________________________________

    def run_model(self):

        """This is where the whole model is ran."""

        if not os.path.exists(os.path.join(self.checkpoint_dir, 'history')):
            os.makedirs(os.path.join(self.checkpoint_dir, 'history'))
        cvscore = []
        test_score = []
        ## Data Preprocess stage ##
        train_data, train_label = self.data_preprocess(self.data,SNR=self.SNR)
        test, test_label = self.data_preprocess(self.test_data, SNR=self.SNR)
        index = list(range(len(train_data)))
        np.random.shuffle(index)
        train_data_shu, train_label_shu = train_data[index],train_label[index]

        test_label = keras.utils.to_categorical(test_label, num_classes=2, dtype='float32')

        np.save(os.path.join(self.checkpoint_dir, 'test_data'), test)
        np.savetxt(os.path.join(self.checkpoint_dir, 'test_label.txt'), test_label)
        ## Cross validation ##
        for i in range(self.CV_num):
            ## prepare call_backs, csv_logger for progress monitoring. ##
            csv_logger = keras.callbacks.CSVLogger(
                os.path.join(self.checkpoint_dir, 'history/training_{}.log'.format(i)))
            self.callbacks = [
                EarlyStopping(
            # look at the validation loss, if it starts increasing, then wait until 20 epochs and if it still doesnt decrease then stop it
                    monitor='val_loss',
                    patience=20,
                    mode='min',
                    verbose=1),
                ModelCheckpoint(os.path.join(self.checkpoint_dir, 'ckt/checkpt_{}.h5'.format(i)),
                                monitor='val_loss',
                                save_best_only=True,
                                mode='min',
                                verbose=1), csv_logger
            ]

            X_train, X_valid, y_train, y_valid = train_test_split(train_data_shu, train_label_shu, test_size=self.valid_size,
                                                                  shuffle=True)
            y_train = keras.utils.to_categorical(y_train, num_classes=2, dtype='float32')
            y_valid = keras.utils.to_categorical(y_valid, num_classes=2, dtype='float32')

            self.model_fn(self.model_type)
            
            ## Training Phase ##
            history = self.model.fit(X_train.reshape(-1, self.img_rows, self.img_cols, self.c_dim), y_train,
                                     batch_size=self.batch_size,
                                     epochs=self.epochs,
                                     verbose=1,
                                     validation_data=(X_valid.reshape(-1, self.img_rows, self.img_cols, self.c_dim), y_valid), shuffle=True,
                                     callbacks=self.callbacks)
            score = self.model.evaluate(X_valid.reshape(-1, self.img_rows, self.img_cols, self.c_dim), y_valid, verbose=0)
            cvscore.append(score[1])
            # Save the model as png file
         #   plot_model(model_fn, to_file='model.png')


        # Plot training & validation accuracy values
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model 1111accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig(os.path.join(self.checkpoint_dir, 'accuracy_history.png'))
            
            # Plot training & validation loss values
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig(os.path.join(self.checkpoint_dir, 'loss_history.png'))



            ## Test Phase ##
            keras.backend.clear_session()
            model = load_model(os.path.join(self.checkpoint_dir, 'ckt/checkpt_{}.h5'.format(i)))
            pred = model.evaluate(test.reshape(-1, self.img_rows, self.img_cols, self.c_dim), test_label)
            test_score.append(pred[1])
            pred = model.predict(test.reshape(-1,self.img_rows, self.img_cols, self.c_dim))
            pos_index = np.where(pred[:, 1] > 0.9)[0]
            ##for k in range(5):
            ##    heatmap = self.return_heatmap(model, test[pos_index[k]])
            ##    diff = test[pos_index[k]] - test[pos_index[k] + int(len(test) / 2)]
            ##    self.plot_heatmap(heatmap,diff.reshape(self.img_rows,self.img_cols), k, i)


#     # Show the model in ipython notebook
#     figure = SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
#     display(figure)
#    
#     # Save the model as png file
#     plot_model(model, to_file='model.png')


            keras.backend.clear_session()

        final_score = np.array([np.mean(cvscore), np.std(cvscore)])
        final_test_score = np.array([np.mean(test_score), np.std(test_score)])
        np.savetxt(os.path.join(self.checkpoint_dir, 'CV_result.txt'), final_score) #result of cross validation 
        np.savetxt(os.path.join(self.checkpoint_dir, 'CV_history.txt'), np.array(cvscore))
        np.savetxt(os.path.join(self.checkpoint_dir, 'test_history.txt'), np.array(test_score))
        np.savetxt(os.path.join(self.checkpoint_dir, 'test_result.txt'), final_test_score) #prediction on new data
        
#___________________________________________________________________________________________________
#___________________________________________________________________________________________________


    # Hyper parameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='datapath') # NB: you add the -- to make it optional
    parser.add_argument('--testpath', type=str, default='test')
    parser.add_argument('--psfpath', type=str, default='psf')
    parser.add_argument('--num_layers',type=int, default=1)
    parser.add_argument('--checkpt', type=str, default='checkpt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=4) # 38 is the value at which it early-stops 
    parser.add_argument('--droprate', type=float, default=0.35)
    parser.add_argument('--num_features', type=int, default=8)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--cv_num', type=int, default=1) # how many tests runs will be repeated. try with 5 or 10
    parser.add_argument('--dense_unit', type=int, default=256)
    parser.add_argument('--SNR', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='customised_CNN')
    parser.add_argument('--c_ratio', nargs='+', type=float) #a priori do not change unless you want to make objects brighter



    
#({{uniform(0, 1)}})
    
    # # Show the model in ipython notebook
    # figure = SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
    # display(figure)
    
    # # Save the model as png file
    # plot_model(model, to_file='model.png')
    


    args = parser.parse_args()
    DLmodel = CNN(datapath=args.datapath, batch_size=args.batch_size, epochs=args.epochs, droprate=args.droprate,
                  num_features=args.num_features, num_class=args.num_class, lr=args.lr,
                   checkpoint_dir=args.checkpt, CV_num=args.cv_num,
                  dense_unit=args.dense_unit, model_type=args.model_type, testpath=args.testpath,
                  psfpath=args.psfpath,decay=args.decay,SNR=args.SNR,num_layers=args.num_layers)
