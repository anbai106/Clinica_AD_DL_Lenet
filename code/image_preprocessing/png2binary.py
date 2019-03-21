#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.misc import imread
import os, h5py
import struct
import glob
import numpy as np
import os
import pandas as pd
from os.path import basename
from array import array
import pickle
from sklearn.model_selection import StratifiedKFold

def load_data_lables(filename, binary=False):
  """ load single batch of cifar """

  width = 145
  depth = 121
  with open(filename, 'rb') as f:
    if binary  == False: ### if the data is stored use pickle as a dictionary
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        participant_id = datadict['participant_id']
    else: ## if the data is stored as binary
        data = f.read()
        data_array = np.frombuffer(data, dtype='<f4')
        print 'Shape:', data_array.shape
        num_pngs = data_array.shape[0] / (width * depth + 1)
        per_png_lable_img = width * depth * 1 + 1
        try:
            isinstance(num_pngs, int)
        except TypeError:
            raise
        X = np.zeros((num_pngs, width, depth, 1))
        Y = np.zeros((num_pngs,))
        #convert the one-dimension array back into label and image format
        for n in xrange(num_pngs):
            Y[n] = int(data_array[per_png_lable_img * n])
            X[n,:,:,:] = data_array[per_png_lable_img * n + 1: per_png_lable_img * (n + 1)].reshape((width, depth, 1))

        print('The label looks like this:\n')
        print(np.array2string(Y))

        ## plot one image as example for sanity check
        # index_img = np.random.randint(0, num_pngs)
        #imshow(np.reshape(X[index_img,:], (145, 121)))

    ### The information for training, validation and test dataset.
    print('This dataset: \n')
    print('In total, we have %d pngs\n' % Y.shape[0])
    print('In total, we have %d CN\n' % int(Y.sum()))

    return X, Y

def dump_train_test_datasets_with_pickle_CN_AD(tsv_diagnosis, png_dir, subject_level=True):
    """
    This is a function to store the data into cifar10 like binary file using PNG-2CIFAR10 package and sklearn StratifiedKFold strategy to
    create the training dataset and test dataset.
    We label AD as 0 and CN as 1.

    Here, we creat 5-fold split data for training, testing and validation dataset in a binary file, also, the corresponding tsv file will be created
    for subject-level diagnoses.

    :param tsv_diagnosis: path to the tsv for CN vs AD with labels
    :param training_portion: the proportion used for training.
    :param pickle_output_dir: the path to the image_preprocessing_output directory.
    :return:
    """

    data_pd = pd.io.parsers.read_csv(tsv_diagnosis, sep='\t')
    if (list(data_pd.columns.values)[0] != 'participant_id') and (list(data_pd.columns.values)[1] != 'session_id')  and (list(data_pd.columns.values)[1] != 'diagnosis'):
        raise Exception('Subjects and visits file is not in the correct format.')
    X = data_pd.participant_id
    y = data_pd.diagnosis

    n_fold = 5

    #####################################################
    ### For subject-level split
    #####################################################
    if subject_level == True:
        for fold_index in range(1, 5):

            skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)
            train_id = [''] * n_fold
            test_id = [''] * n_fold
            a = 0

            for train_index, test_index in skf.split(X, y):
                print("SPLIT iteration:", a + 1, "Traing:", train_index, "Test", test_index)
                train_id[a] = train_index
                test_id[a] = test_index
                a = a + 1

            testid = test_id[fold_index]
            trainid = train_id[fold_index]
            x_training = X[trainid]
            y_training = y[trainid]

            x_test_val = X[testid].reset_index(drop=True)
            y_test_val = y[testid].reset_index(drop=True)
            skf_2 = StratifiedKFold(2, shuffle=False, random_state=0)
            for test_ind, valid_ind in skf_2.split(x_test_val, y_test_val):
                print("SPLIT iteration:", "Test:", test_ind, "Validation", valid_ind)

            x_val = x_test_val[valid_ind]
            y_val = y_test_val[valid_ind]
            x_test = x_test_val[test_ind]
            y_test = y_test_val[test_ind]

            if x_training.empty or x_test.empty or x_val.empty:
                print('Dataframe is empty exit!!')
                exit()

            print('For fold %d \n' % fold_index)
            print('Training dataset: \n')
            print('In total, we have %d CN\n' % int(y_training.sum()))
            print('test dataset: \n')
            print('In total, we have %d CN\n' % int(y_test.sum()))
            print('Validation dataset: \n')
            print('In total, we have %d CN\n' % int(y_val.sum()))

            ######## create the variables for using
            training_num_slices = 0
            test_num_slices = 0
            val_num_slices = 0
            subject_list_training = []
            label_training = []
            subject_list_test = []
            label_test = []
            subject_list_val = []
            label_val = []


            data_training = array('f')
            data_test = array('f')
            data_val = array('f')

            ##Training
            for i in range(y_training.shape[0]):
                imgs = glob.glob(os.path.join(png_dir, 'sliced_image', x_training.iloc[i] + "-c1-frame" + "*-slice*.h5"))
                try:
                    imgs[0]
                except IndexError:
                    print 'Empty list'
                    exit()
                for img in imgs:
                    print "Processing for this h5: %s" % basename(img)

                    h5f = h5py.File(img, 'r')
                    im = h5f['image'][:]
                    h5f.close()

                    ### sanity check
                    print 'Max and min value for each image %f and %f' % (np.max(im), np.min(im))

                    width, height = im.shape[0], im.shape[1]
                    print 'image size is: width: %d, height, %d' % (width, height)

                    data_training.append(y_training.iloc[i])

                    #then write the rows
                    for k in range(0, width):
                        for j in range(0, height):
                            data_training.append(im[k, j])

                    training_num_slices += 1
                    subject_list_training.extend([x_training.iloc[i]] * len(imgs))
                    label_training.extend([y_training.iloc[i]] * len(imgs))

            ## write the binary file
            output_file_training = open(os.path.join(png_dir, 'sliced_image', 'subject_level_training_adni_AD_vs_CN_baseline_T1_slices_' + str(training_num_slices) + '_fold_' + str(fold_index) + '.bin'), 'wb')
            data_training.tofile(output_file_training)
            output_file_training.close()
            ## write the participant_tsv file
            tsv_dic_training = {'participant_id': subject_list_training, 'diagnosis': label_training}
            tsv_df_training = pd.DataFrame.from_dict(tsv_dic_training)
            tsv_df_training.to_csv(os.path.join(png_dir, 'sliced_image', 'subject_level_training_adni_AD_vs_CN_baseline_T1_slices_' + str(training_num_slices) + '_fold_' + str(fold_index) + '.tsv'),
                               index=False, sep='\t', encoding='utf-8')

            ##Test
            for i in range(y_test.shape[0]):
                imgs = glob.glob(os.path.join(png_dir, 'sliced_image', x_test.iloc[i] + "-c1-frame" + "*-slice*.h5"))
                try:
                    imgs[0]
                except IndexError:
                    print 'Empty list'
                    exit()
                for img in imgs:
                    print "Processing for this h5: %s" % basename(img)

                    h5f = h5py.File(img, 'r')
                    im = h5f['image'][:]
                    h5f.close()

                    ### sanity check
                    print 'Max and min value for each image %f and %f' % (np.max(im), np.min(im))

                    width, height = im.shape[0], im.shape[1]
                    print 'image size is: width: %d, height, %d' % (width, height)

                    data_test.append(y_test.iloc[i])

                    #then write the rows
                    for k in range(0, width):
                        for j in range(0, height):
                            data_test.append(im[k, j])

                    test_num_slices += 1
                    subject_list_test.extend([x_test.iloc[i]] * len(imgs))
                    label_test.extend([y_test.iloc[i]] * len(imgs))

            ## write the binary file
            output_file_test = open(os.path.join(png_dir, 'sliced_image', 'subject_level_test_adni_AD_vs_CN_baseline_T1_slices_' + str(test_num_slices) + '_fold_' + str(fold_index) + '.bin'), 'wb')
            data_test.tofile(output_file_test)
            output_file_test.close()
            ## write the participant_tsv file
            tsv_dic_test = {'participant_id': subject_list_test, 'diagnosis': label_test}
            tsv_df_test = pd.DataFrame.from_dict(tsv_dic_test)
            tsv_df_test.to_csv(os.path.join(png_dir, 'sliced_image', 'subject_level_test_adni_AD_vs_CN_baseline_T1_slices_' + str(test_num_slices) + '_fold_' + str(fold_index) + '.tsv'),
                               index=False, sep='\t', encoding='utf-8')

            ##val
            for i in range(y_val.shape[0]):
                imgs = glob.glob(os.path.join(png_dir, 'sliced_image', x_val.iloc[i] + "-c1-frame" + "*-slice*.h5"))
                try:
                    imgs[0]
                except IndexError:
                    print 'Empty list'
                    exit()
                for img in imgs:
                    print "Processing for this h5: %s" % basename(img)

                    h5f = h5py.File(img, 'r')
                    im = h5f['image'][:]
                    h5f.close()

                    ### sanity check
                    print 'Max and min value for each image %f and %f' % (np.max(im), np.min(im))

                    width, height = im.shape[0], im.shape[1]
                    print 'image size is: width: %d, height, %d' % (width, height)

                    data_val.append(y_val.iloc[i])

                    #then write the rows
                    for k in range(0, width):
                        for j in range(0, height):
                            data_val.append(im[k, j])

                    val_num_slices += 1
                    subject_list_val.extend([x_val.iloc[i]] * len(imgs))
                    label_val.extend([y_val.iloc[i]] * len(imgs))

            ## write the binary file
            output_file_val = open(os.path.join(png_dir, 'sliced_image', 'subject_level_val_adni_AD_vs_CN_baseline_T1_slices_' + str(val_num_slices) + '_fold_' + str(fold_index) + '.bin'), 'wb')
            data_val.tofile(output_file_val)
            output_file_val.close()
            ## write the participant_tsv file
            tsv_dic_val = {'participant_id': subject_list_val, 'diagnosis': label_val}
            tsv_df_val = pd.DataFrame.from_dict(tsv_dic_val)
            tsv_df_val.to_csv(os.path.join(png_dir, 'sliced_image', 'subject_level_val_adni_AD_vs_CN_baseline_T1_slices_' + str(val_num_slices) + '_fold_' + str(fold_index) + '.tsv'),
                               index=False, sep='\t', encoding='utf-8')

    #####################################################
    ### For slice-level split, we can read the subject-level binary files and resplit it with slice-level.
    #####################################################

    else:
        filename = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/deep_learning_classification/image_preprocessing_output/ADNI_baseline_t1/sliced_image/subject_level_test_adni_AD_vs_CN_baseline_T1_slices_6998_fold_0.bin'
        x_test, y_test = load_data_lables(filename, binary=True)
        filename = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/deep_learning_classification/image_preprocessing_output/ADNI_baseline_t1/sliced_image/subject_level_training_adni_AD_vs_CN_baseline_T1_slices_55234_fold_0.bin'
        x_train, y_train = load_data_lables(filename, binary=True)
        filename = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/deep_learning_classification/image_preprocessing_output/ADNI_baseline_t1/sliced_image/subject_level_val_adni_AD_vs_CN_baseline_T1_slices_7009_fold_0.bin'
        x_valid, y_valid = load_data_lables(filename, binary=True)

        skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)
        train_id = [''] * n_fold
        test_id = [''] * n_fold
        a = 0

        for train_index, test_index in skf.split(np.concatenate((x_train, x_valid, x_test)),
                                                 np.concatenate((y_train, y_valid, y_test))):
            train_id[a] = train_index
            test_id[a] = test_index
            a = a + 1

        for fi in range(n_fold):

            testid = test_id[fi]
            trainid = train_id[fi]
            # xx_train = np.concatenate((x_train, x_valid, x_test))[trainid]
            # yy_train = np.concatenate((y_train, y_valid, y_test))[trainid]

            skf_2 = StratifiedKFold(2, shuffle=False, random_state=0)
            for test_ind, valid_ind in skf_2.split(np.concatenate((x_train, x_valid, x_test))[testid],
                                                   np.concatenate((y_train, y_valid, y_test))[testid]):
                print("SPLIT iteration:", "Test:", test_ind, "Validation", valid_ind)

            # xx_valid = np.concatenate((x_train, x_valid, x_test))[testid][valid_ind]
            # yy_valid = np.concatenate((y_train, y_valid, y_test))[testid][valid_ind]
            # xx_test = np.concatenate((x_train, x_valid, x_test))[testid][test_ind]
            # yy_test = np.concatenate((y_train, y_valid, y_test))[testid][test_ind]

            ### use pickle to store the dataframe
            dic_training = {"data": np.concatenate((x_train, x_valid, x_test))[trainid], "label": np.concatenate((y_train, y_valid, y_test))[trainid]}
            pickle.dump(dic_training, open(os.path.join(png_dir, 'sliced_image', 'slice_level_training_adni_AD_vs_CN_baseline_T1_slices_' + str(np.concatenate((x_train, x_valid, x_test))[trainid].shape[0]) + '_fold_' + str(fi) + '.bin'), "wb"))
            dic_test = {"data": np.concatenate((x_train, x_valid, x_test))[testid][test_ind], "label": np.concatenate((y_train, y_valid, y_test))[testid][test_ind]}
            pickle.dump(dic_test, open(os.path.join(png_dir, 'sliced_image', 'slice_level_test_adni_AD_vs_CN_baseline_T1_slices_' + str(np.concatenate((x_train, x_valid, x_test))[testid][test_ind].shape[0]) + '_fold_' + str(fi) + '.bin'), "wb"))
            dic_valid = {"data": np.concatenate((x_train, x_valid, x_test))[testid][valid_ind], "label": np.concatenate((y_train, y_valid, y_test))[testid][valid_ind]}
            pickle.dump(dic_valid, open(os.path.join(png_dir, 'sliced_image', 'slice_level_valid_adni_AD_vs_CN_baseline_T1_slices_' + str(np.concatenate((x_train, x_valid, x_test))[testid][valid_ind].shape[0]) + '_fold_' + str(fi) + '.bin'), "wb"))

    print "finish!!!"

## to test if the png images were well organised in the binary file
# filename = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/deep_learning_classification/image_preprocessing_output/ADNI_baseline_t1/sliced_image/subject_level_test_adni_AD_vs_CN_baseline_T1_slices_6998_fold_0.bin'
# X, y = load_data_lables(filename, binary=True)