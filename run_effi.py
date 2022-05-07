from __future__ import print_function
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate
from keras.optimizers import Adam, SGD, Nadam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.utils import Sequence #, multi_gpu_model

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as in_pi
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as re_pi
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vg_pi
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xc_pi
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as inr_pi
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input as de_pi
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as mo_pi
import efficientnet.keras as efn 

import sys, os, six, time, copy, random
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

#from skimage.io import imread
import cv2
from cv2 import imread

from scipy.ndimage import zoom



nb_classes = 3

COVIDX_D = "../covidx_ver5_dataset-train/n1024_dataset/"
COVIDX_CSV = "../covidx_ver5_dataset-train/train_COVIDx5.csv"
MYCOVID_D = "../my_covid-train/n1024_dataset/"
MYCOVID_CSV = "../my_covid-train/my_tr.csv"
HYOGO_D = "../00_all_hospitals_splitted/n1024_dataset/"
HYOGO_CSV = "../00_all_hospitals_splitted/my_all.csv"



def split_df_for_metric(df, N, args):
    dd = df.copy(deep=True)
    NN = int(N/3)

    d0 = df[df.iloc[:,1]==0].sample(n=NN, random_state=args.seed)
    d1 = df[df.iloc[:,1]==1].sample(n=NN, random_state=args.seed)
    d2 = df[df.iloc[:,1]==2].sample(n=NN, random_state=args.seed)
    
    if args.verbose == 1:
        print("normal:", d0)
        print("pneumonia:", d1)
        print("covid", d2)
    
    dd = dd.drop(d0.index)
    dd = dd.drop(d1.index)
    dd = dd.drop(d2.index)
    
    df1 = dd
    df2 = pd.concat([d0,d1,d2], axis=0)
    print("data splitting:", len(df1), len(df2))
    return df1, df2



###############
## Generator ##
###############

def idg_func(params, batch_X, BB):
    NN = random.randint(0, 1000000)    

    datagen = ImageDataGenerator(**params)
    idg_X = datagen.flow(batch_X, batch_size=BB, shuffle=False, seed=NN).__next__()
    return idg_X


def mix_up(X1, Y1, X2, Y2, alpha, batch_size):
    if X1.shape[0] != X2.shape[0]:
        import pdb; pdb.set_trace()

    assert X1.shape[0] == X2.shape[0]
    assert Y1.shape[0] == Y2.shape[0]
    batch_size = X1.shape[0]
    l = np.random.beta(alpha, alpha, batch_size)
    X_l = l.reshape(batch_size, 1, 1, 1)
    Y_l = l.reshape(batch_size, 1)
    X = X1 * X_l + X2 * (1 - X_l)
    Y = Y1 * Y_l + Y2 * (1 - Y_l)
    return X, Y


def ricap(image_batch, label_batch, beta=0.3, use_same_random_value_on_batch=False):
    assert image_batch.shape[0] == label_batch.shape[0]
    assert image_batch.ndim == 4
    batch_size, image_y, image_x = image_batch.shape[:3]

    if use_same_random_value_on_batch:
        w_dash = np.random.beta(beta, beta) * np.ones(batch_size)
        h_dash = np.random.beta(beta, beta) * np.ones(batch_size)
    else:
        w_dash = np.random.beta(beta, beta, size=(batch_size))
        h_dash = np.random.beta(beta, beta, size=(batch_size))
    w = np.round(w_dash * image_x).astype(np.int32)
    h = np.round(h_dash * image_y).astype(np.int32)

    output_images = np.zeros(image_batch.shape)
    output_labels = np.zeros(label_batch.shape)

    def create_masks(start_xs, start_ys, end_xs, end_ys):
        mask_x = np.logical_and(np.arange(image_x).reshape(1, 1, -1, 1) >= start_xs.reshape(-1, 1, 1, 1),
                                np.arange(image_x).reshape(1, 1, -1, 1) < end_xs.reshape(-1, 1, 1, 1))
        mask_y = np.logical_and(np.arange(image_y).reshape(1, -1, 1, 1) >= start_ys.reshape(-1, 1, 1, 1),
                                np.arange(image_y).reshape(1, -1, 1, 1) < end_ys.reshape(-1, 1, 1, 1))
        mask = np.logical_and(mask_y, mask_x)
        mask = np.logical_and(mask, np.repeat(True, image_batch.shape[3]).reshape(1, 1, 1, -1))
        return mask

    def crop_concatenate(wk, hk, start_x, start_y, end_x, end_y):
        nonlocal output_images, output_labels
        xk = (np.random.rand(batch_size) * (image_x - wk)).astype(np.int32)
        yk = (np.random.rand(batch_size) * (image_y - hk)).astype(np.int32)
        target_indices = np.arange(batch_size)
        np.random.shuffle(target_indices)
        weights = wk * hk / image_x / image_y

        dest_mask = create_masks(start_x, start_y, end_x, end_y)
        target_mask = create_masks(xk, yk, xk + wk, yk + hk)

        output_images[dest_mask] = image_batch[target_indices][target_mask]
        output_labels += weights.reshape(-1, 1) * label_batch[target_indices]

    # left-top crop
    crop_concatenate(w, h,
                     np.repeat(0, batch_size), np.repeat(0, batch_size),
                     w, h)
    # right-top crop
    crop_concatenate(image_x - w, h,
                     w, np.repeat(0, batch_size),
                     np.repeat(image_x, batch_size), h)
    # left-bottom crop
    crop_concatenate(w, image_y - h,
                     np.repeat(0, batch_size), h,
                     w, np.repeat(image_y, batch_size))
    # right-bottom crop
    crop_concatenate(image_x - w, image_y - h,
                     w, h, np.repeat(image_x, batch_size),
                     np.repeat(image_y, batch_size))

    return output_images, output_labels


def change_filepath_of_df(df, folder):
    df2 = df.copy()
    N = len(df2)
    for i in range(N):
        df2.iloc[i,0] = folder + "/" + df.iloc[i,0].strip() 
        
    return df2


class MyGenerator(Sequence):

    def __init__(self, main_df, ad_df=None, ad_ratio=0, batch_size=1, width=512, height=512, ch=1, aug=0, alpha=0, beta=0, shuffle=True):
        self.main_df = main_df.copy()
        N = len(main_df)

        if ad_df is None:
            self.ad_df = None
            self.length = N 
        else:
            self.ad_df = ad_df.copy()
            NNN = len(ad_df)
            self.length = int((1+ad_ratio)*N)
                    
        self.ricap_beta = beta
        self.mixup_alpha = alpha
        self.aug = aug
        
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.ch = ch
        self.shuffle = shuffle
        self.num_batches_per_epoch = int((self.length - 1) / batch_size) + 1

        if aug == 0:
            params = {
            }
        elif aug == 1:
            params = {
                'rotation_range': 2,
                "width_shift_range": 0.02,
                "height_shift_range":0.02,
                "zoom_range": 0.02,
            }       
        elif aug == 2:
            params = {
                'rotation_range': 5,
                "width_shift_range": 0.05,
                "height_shift_range":0.05,
                "zoom_range": 0.05,
            }
        elif aug == 3:
            params = {
                'horizontal_flip': True,
                'rotation_range': 5,
                "width_shift_range": 0.05,
                "height_shift_range":0.05,
                "shear_range": 0.05,
                "zoom_range": 0.05,
            }
        elif aug == 4:
            params = {
                'horizontal_flip': True,
                'rotation_range': 10,
                "width_shift_range": 0.1,
                "height_shift_range":0.1,
                "shear_range": 0.1,
                "zoom_range": 0.1,
            }
        elif aug == 5:
            params = {
                'horizontal_flip': True,
                'rotation_range': 15,
                "width_shift_range": 0.15,
                "height_shift_range":0.15,
                "shear_range": 0.15,
                "zoom_range": 0.15,
            }
        else:
            print("argument error")
            exit()

        if aug > 0:
            self.idg_params = params

        self.on_epoch_end()

        return

    def __getitem__(self, idx):
        batch_X, batch_Y = self.__load(idx)

        if self.aug > 0:
            batch_X = idg_func(self.idg_params, batch_X, self.batch_size)

        if self.mixup_alpha > 0:
            batch_X_2, batch_Y_2 = self.__load(idx + 1)
                    
            m1, m2 = batch_X.shape[0], batch_X_2.shape[0]

            if m1 == 0:
                batch_X, batch_Y = self.__load(0)
                m1 = batch_X.shape[0]
            elif m2 == 0:  
                batch_X_2, batch_Y_2 = self.__load(0)
                m2 = batch_X_2.shape[0]

            if m1 > m2:
                batch_X = batch_X_2[:m2]
                batch_Y = batch_Y_2[:m2]
            elif m2 > m1:
                batch_X_2 = batch_X_2[:m1]
                batch_Y_2 = batch_Y_2[:m1]

            if self.aug > 0:
                batch_X_2 = idg_func(self.idg_params, batch_X_2, self.batch_size)

            batch_X, batch_Y = mix_up(batch_X, batch_Y, batch_X_2, batch_Y_2, self.mixup_alpha, self.batch_size)

        if self.ricap_beta > 0:
            batch_X, batch_Y = ricap(batch_X, batch_Y, self.ricap_beta)

        return batch_X, batch_Y

    def __load(self, idx):
        NNN = self.num_batches_per_epoch
        idx = idx % NNN

        self.load_times += 1
        if self.load_times == NNN:
            self.on_epoch_end()
            
        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.length:
            end_pos = self.length
        item_indices = self.indices[start_pos:end_pos]

        df = self.df
        N = len(item_indices)
        imgs = np.empty((N, self.height, self.width, self.ch), dtype=np.float32)
        labels = np.empty((N, nb_classes), dtype=np.float32)

        for i, idx in enumerate(item_indices):
            d = df.iloc[idx]
            item_path = d.iloc[0]
            img = imread(item_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print("file not found:", item_path)
                exit()
                
            if len(img.shape) == 3:
                img = img[:,:,0]

            w = img.shape[1]
            h = img.shape[0]
            r = (self.height/float(h), self.width/float(w))
            img = zoom(img, r, order=1)

            if img.max() > 10:
                img = (img / 255).astype(np.float32)
                
            for j in range(3):
                imgs[i,:,:,j] = img
                
            l = d.iloc[1]
            label = np.zeros( (3,) )
            if l == 0:
                label[0] = 1
            elif l == 1:
                label[1] = 1
            elif l == 2:
                label[2] = 1
            else:
                print("error")
                exit()
                
            labels[i,:] = label

        return imgs, labels

    def __len__(self):
        if self.shuffle:            
            return 1000000
        else:
            return self.num_batches_per_epoch
        
    def on_epoch_end(self):
        self.load_times = 0
        
        if self.ad_df is None:
            self.df = self.main_df
        else:
            NN = self.length - len(self.main_df)
            tmp_df = self.ad_df.sample(n=NN)
            df = pd.concat([tmp_df, self.main_df], axis=0)             
            self.df = df 

        self.indices = list(range(len(self.df)))
        if self.shuffle > 0:
            N = random.randint(0, 1000000)
            random.seed(N)
            random.shuffle(self.indices)

        return
    

##################
# Model
##################   
def build_model(input_shape, args):
    D = args.d
    F = args.f
    V = args.v

    input_tensor = Input(shape=input_shape)

    if args.tf == "in":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
        #pi = in_pi
    elif args.tf == "inr":
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
        #pi = inr_pi
    elif args.tf == "vg":
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        #pi = vg_pi
    elif args.tf == "xc":
        base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
        #pi = xc_pi
    elif args.tf == "re":
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
        #pi = re_pi
    elif args.tf == "de":
        base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=input_tensor)
        #pi = de_pi
    elif args.tf == "mo":
        base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=input_tensor)
        #pi = mo_pi
    elif args.tf.find("nef") > -1:
        print("Noisy Student")
        if args.tf == "nef0":
            base_model = efn.EfficientNetB0(weights='noisy-student', include_top=False, input_tensor=input_tensor)
        elif args.tf == "nef1":
            base_model = efn.EfficientNetB1(weights='noisy-student', include_top=False, input_tensor=input_tensor)
        elif args.tf == "nef2":
            base_model = efn.EfficientNetB2(weights='noisy-student', include_top=False, input_tensor=input_tensor)
        elif args.tf == "nef3":
            base_model = efn.EfficientNetB3(weights='noisy-student', include_top=False, input_tensor=input_tensor)
        elif args.tf == "nef4":
            base_model = efn.EfficientNetB4(weights='noisy-student', include_top=False, input_tensor=input_tensor)
        elif args.tf == "nef5":
            base_model = efn.EfficientNetB5(weights='noisy-student', include_top=False, input_tensor=input_tensor)
        elif args.tf == "nef6":
            base_model = efn.EfficientNetB6(weights='noisy-student', include_top=False, input_tensor=input_tensor)
        elif args.tf == "nef7":
            base_model = efn.EfficientNetB7(weights='noisy-student', include_top=False, input_tensor=input_tensor)
        else:
            print("unknown network type:", args.tf)
            exit()
    elif args.tf.find("ef") > -1:
        if args.tf == "ef0":
            base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif args.tf == "ef1":
            base_model = efn.EfficientNetB1(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif args.tf == "ef2":
            base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif args.tf == "ef3":
            base_model = efn.EfficientNetB3(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif args.tf == "ef4":
            base_model = efn.EfficientNetB4(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif args.tf == "ef5":
            base_model = efn.EfficientNetB5(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif args.tf == "ef6":
            base_model = efn.EfficientNetB6(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif args.tf == "ef7":
            base_model = efn.EfficientNetB7(weights='imagenet', include_top=False, input_tensor=input_tensor)
        else:
            print("unknown network type:", args.tf)
            exit()
    else:
        print("unknown network type:", args.tf)
        exit()

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(F, activation='relu')(x)
    if D > 0:
        x = Dropout(D)(x)
 
    pred = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=pred)

    layer_num = len(base_model.layers)
    for layer in base_model.layers[:int(layer_num * V)]:
        layer.trainable = False

    return model 



def train_val_test(args):
    B = args.b
    E = args.e
    Z = args.z
    R = args.r
    DLR = args.dlr
    L = args.l

    print("****************************" * 2)
    print("****************************" * 2)
    print("building model ...")
    L = args.l
    input_shape = (L, L, 3)
    model = build_model(input_shape, args)


    
    print("****************************" * 2)
    print("****************************" * 2)
    
    tr_d = HYOGO_D
    val_d = HYOGO_D
    df = pd.read_csv(HYOGO_CSV, header=None)
    df = change_filepath_of_df(df, tr_d)
    tr, val = split_df_for_metric(df, 90, args)
    
    if args.ad == 0:
        ad_df = None
        ad_d = None

        te_d1 = COVIDX_D
        df1 = pd.read_csv(COVIDX_CSV, header=None)
        df1 = change_filepath_of_df(df1, te_d1)
        _, te1 = split_df_for_metric(df1, 300, args)
        
        te_d2 = MYCOVID_D
        df2 = pd.read_csv(MYCOVID_CSV, header=None)
        df2 = change_filepath_of_df(df2, te_d2)
        _, te2 = split_df_for_metric(df2, 300, args)
    elif args.ad == 1:
        te_d1 = COVIDX_D
        df1 = pd.read_csv(COVIDX_CSV, header=None)
        df1 = change_filepath_of_df(df1, te_d1)
        ad_df, te1 = split_df_for_metric(df1, 300, args)
        ad_d = COVIDX_D
        
        te_d2 = MYCOVID_D
        df2 = pd.read_csv(MYCOVID_CSV, header=None)
        df2 = change_filepath_of_df(df2, te_d2)
        _, te2 = split_df_for_metric(df2, 300, args)                
    elif args.ad == 2:
        te_d1 = COVIDX_D
        df1 = pd.read_csv(COVIDX_CSV, header=None)
        df1 = change_filepath_of_df(df1, te_d1)
        _, te1 = split_df_for_metric(df1, 300, args)
                
        te_d2 = MYCOVID_D
        df2 = pd.read_csv(MYCOVID_CSV, header=None)
        df2 = change_filepath_of_df(df2, te_d2)
        ad_df, te2 = split_df_for_metric(df2, 300, args)
        ad_d = MYCOVID_D
    elif args.ad == 3:
        te_d1 = COVIDX_D
        df1 = pd.read_csv(COVIDX_CSV, header=None)
        df1 = change_filepath_of_df(df1, te_d1)
        ad_df1, te1 = split_df_for_metric(df1, 300, args)
                
        te_d2 = MYCOVID_D
        df2 = pd.read_csv(MYCOVID_CSV, header=None)
        df2 = change_filepath_of_df(df2, te_d2)
        ad_df2, te2 = split_df_for_metric(df2, 300, args)

        ad_df = pd.concat([ad_df1, ad_df2], axis=0)
        ad_d = "combined"        
    else:
        print("argument error in args.ad:", args.ad)
        exit()
        
    print("train:", tr_d)
    print("val1:", val_d)
    print("val2:", te_d1)
    print("val3:", te_d2)
    print("additional dataet:", ad_d)

    print("train:", len(tr))
    print("val1:", len(val))
    print("val2:", len(te1))
    print("val3:", len(te2))
    if ad_df is None:
        print("additional dataet:", 0)
    else:
        print("additional dataet:", len(ad_df))

    tr_gen = MyGenerator(tr, ad_df=ad_df, ad_ratio=args.ad_ratio, batch_size=B, width=L, height=L, ch=3, aug=args.aug, alpha=args.alpha, beta=args.beta, shuffle=True)
    val_gen = MyGenerator(val, batch_size=1, width=L, height=L, ch=3, shuffle=False)
    te_gen1 = MyGenerator(te1, batch_size=1, width=L, height=L, ch=3, shuffle=False)
    te_gen2 = MyGenerator(te2, batch_size=1, width=L, height=L, ch=3, shuffle=False)


    
    if args.op == 0:
        o = SGD(lr=R, momentum=0.9, nesterov=True)
    elif args.op == 1:
        o = Adam(lr=R)
    elif args.op == 2:
        o = Nadam(lr=R)
    else:
        o = RMSprop(lr=R)
                                        
    model.compile(loss='categorical_crossentropy', optimizer=o, metrics=['accuracy'])

    cs = []
    csv_path = args.o + "log/" + args.sig + ".csv"
    csv = CSVLogger(csv_path)
    cs = cs + [csv]
    if args.es > 0:
        stop = EarlyStopping(monitor='val_loss', patience=args.es)
        cs = cs + [stop]
    if DLR > 0:
        lrs = LearningRateScheduler(lambda ep: float(R / 10 ** (ep * DLR // E)))
        cs = cs + [lrs]
    if args.save > 0:
        path = args.o + "/model/" + "best___" + args.sig + '.h5'
        mcp = ModelCheckpoint(filepath=path,
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=True,
                              save_weights_only=False,
                              mode='min',
                              period=1)
        cs = cs + [mcp]
        
    class_weight = {0: 1.0,
                    1: 1.0,
                    2: args.covid_weight}

    hist = model.fit_generator(tr_gen, epochs=E, steps_per_epoch=Z, verbose=args.verbose, validation_data=val_gen, validation_steps=len(val_gen), callbacks=cs, class_weight=class_weight, max_queue_size=64, workers=10)
    
    if args.save > 0:
        print('saving final model ... ')
        path = args.o + "/model/" + "final___" + args.sig + '.h5'
        model.save(path)

    print("evaluating val data ...")
    if args.eval_test > 0:
        targets = [ (val_gen,"val1"), (te_gen1,"val2"), (te_gen2,"val3") ]
    else:
        targets = [ (val_gen,"val1") ]

    losses = []
    vs = []
    for (gen, name) in targets:
        score = model.evaluate_generator(gen, steps=len(gen), verbose=0)
        loss = score[0]
        accu = score[1]
        losses.append(loss)
        vs.append(accu)
        print('Final %s loss, %s, ---, %s' % (name,loss,args.sig))
        print('Final %s accuracy, %s, ---, %s' % (name,accu,args.sig))

    if args.eval_test > 0:
        te_idx = 0
        for te_gen in [val_gen, te_gen1, te_gen2]:
            print("calculating confusion matrix of val data ...")
            preds = []
            truth = []

            i = 0
            N = len(te_gen)
            for ims, labels in te_gen:
                i += 1
                if i > N:
                    break
                p = model.predict(ims, verbose=0)
                preds = preds + p.tolist()
                truth = truth + labels.tolist()
        
            truth = np.array(truth)
            preds = np.array(preds)
    
            if args.save > 0:
                print('saving prediction of val data ... ')
                save_pred(truth, preds, args, te_idx)
            
            truth = np.array(truth).argmax(axis=1)
            preds = np.array(preds).argmax(axis=1)
            print("*** confusion matrix ***")
            print(confusion_matrix(truth, preds))

            te_idx += 1

    return vs



############################################
# Helper
############################################
def save_pred(te_labels, proba_result, args, te_idx):
    te_labels = te_labels.argmax(axis=1)
    predicted_labels = proba_result.argmax(axis=1)
    
    N = proba_result.shape[0]
    t = np.zeros( (N,2+nb_classes) ) 
    t[:,0] = predicted_labels
    t[:,1] = te_labels
    t[:,2:] = proba_result

    print("******************************")
    print("summary of truth in test set")
    print( "total =>", te_labels.shape[0] )
    print( "  0 =>", (te_labels == 0).sum() )
    print( "  1 =>", (te_labels == 1).sum() )
    print( "  2 =>", (te_labels == 2).sum() )

    A = (te_labels == predicted_labels).sum()
    s = args.sig
    final_accu = A/float(N)
    print( "Test Accuracy %f (percent) %d/%d --- %s" % (final_accu*100, A,N, s) )

    h = "# predicted_label, true_label, prob0, prob1, prob2, final_test_accuracy=%s" % final_accu
    path =  args.o + "/pred/" + args.sig + "_te_idx%s.csv" % te_idx
    np.savetxt(path, t, delimiter=",", header=h)

    return 

def save_history(history, args):
    l = history.history['acc']
    N = len(l)

    h = np.zeros( (N,4) )
    h[:,0] = np.array(history.history['loss'])
    h[:,1] = np.array(history.history['acc'])
    h[:,2] = np.array(history.history['val_loss'])
    h[:,3] = np.array(history.history['val_acc'])
    header = "# loss, accuracy, val_loss, val_accuracy"
    
    path = args.o + "/log/history_" + args.sig + ".csv"
    np.savetxt(path, h, delimiter=",", header=header)


    return




####################
# RUN
####################

def run(args):
    SIG = args.sig

    print("start training and validating ...")
    vs = train_val_test(args)
    
    path = args.o + "/all_results.csv"
    print(path)
    f = open(path, "a")
    
    for v in vs:
        f.write(str(v))
        f.write(", ")

    f.write(SIG)
    f.write("\n")
    f.close()
    
    return



#############################
# Main
#############################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str, default="./result/hoge/")  # directory for saving
    
    parser.add_argument("-e", type=int, default=1)    # epoch
    parser.add_argument("-b", type=int, default=2)   # batch
    parser.add_argument("-z", type=int, default=11)  # number of images which are processed per one epoch
    parser.add_argument("-r", type=float, default=0.0001) # learning rate 
    parser.add_argument("--op", type=int, default=3)     # type of optimizer
    parser.add_argument("--dlr", type=int, default=0)
                         
    parser.add_argument("-v", type=float, default=0.5) # ratio of frozen layers 
    parser.add_argument("-f", type=int, default=16)    # FC
    parser.add_argument("-d", type=float, default=0.1) # dropout
            
    parser.add_argument("--tf", type=str, default="vg")
    parser.add_argument("-l", type=int, default=112)   # size of image in training data

    parser.add_argument("--aug", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--es", type=int, default=0)

    parser.add_argument("--covid-weight", type=float, default=1.0)
    parser.add_argument("--ad", type=int, default=0)
    parser.add_argument("--ad-ratio", type=float, default=0)
    
    parser.add_argument("--eval-test", type=int, default=0)

    parser.add_argument("--save", type=int, default=0) # save model
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=int, default=1)

    
    
    args = parser.parse_args()

    sig = "e%s_b%s_z%s_r%s_op%s_dlr%s__v%s_f%s_d%s__aug%s_alpha%s_beta%s__tf%s_l%s_covid_weight%s__ad%s_ad_ratio%s__es%s_seed%s__index%s" % (args.e,args.b,args.z,args.r,args.op,args.dlr, args.v,args.f,args.d, args.aug,args.alpha,args.beta, args.tf,args.l,args.covid_weight, args.ad,args.ad_ratio, args.es,args.seed, args.index)
    args.sig = sig

    args.o = d = args.o + "/"
    for i in [d, d+"/log", d+"/model", d+"/pred" ]:
        os.makedirs(i, exist_ok=True)

    print(args)
    run(args)
