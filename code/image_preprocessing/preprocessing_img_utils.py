#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = "Junhao WEN"
__copyright__ = "Copyright 2016, The Aramis Lab Team"
__credits__ = ["Michael Bacci", "Junhao WEN"]
__license__ = "??"
__version__ = "1.0.0"
__maintainer__ = "Junhao WEN"
__email__ = "junhao.wen@inria.fr"
__status__ = "Development"


def merge_img(tissue_img, output_dir):
    """
    concatenate multiple images in one direction
    :param tissue:
    :return:
    """
    import nibabel as nib
    import numpy as np
    from os.path import join, isfile
    import os

    if not os.path.exists(join(output_dir, 'merged_nifti')):
        os.makedirs(join(output_dir, 'merged_nifti'))

    out_img_path = join(output_dir, 'merged_nifti', 'merged_all_nii.nii')
    if len(tissue_img) == 0:
        raise RuntimeError('The length of the list of tissues must be greater than zero!!!')
    if isfile(out_img_path):
        print "Note: merged_image exists, skip this step!!!"
    else:
        print "Note: merged_image does not exist, concatenate all the 3D images to be a 4D image!!!"
        if not isinstance(tissue_img, basestring):
            img_0 = nib.load(tissue_img[0])
            img_shape = img_0.get_data().shape
            img_shape += (len(tissue_img),)

            data = np.empty(img_shape, dtype=np.float64)

            for i in xrange(len(tissue_img)):
                img = nib.load(tissue_img[i])
                data[..., i] = img.get_data()
            out_img = nib.Nifti1Image(data, img_0.affine, header=img_0.header)
            nib.save(out_img, out_img_path)
        else:
            raise("Not possible, at least you have to have two subjects")

    return out_img_path

def nii_png_med2image(root_path, smoothed_img, output_dir, output_file_type, output_file_stem, z_dimensions):
    """
    Convert a concatenated 4D nifti image to a 2D grayscale png image:
    using med2img to concert the 4D nifti into a rgb image; then convert them into a grayscale image, and delete the empty-intensity image
    :param root_path: the path to med2img package
    :param smoothed_img: the concatenated smoothed 4D nifti
    :param output_dir:
    :param output_file_type:
    :param output_file_stem:
    :param z_dimensions: the third dimension of the concatenated nifti image
    :return:
    """
    from os import system, listdir, makedirs, remove
    from os.path import join, exists
    import nibabel as nib
    import errno
    from scipy.ndimage import imread
    from scipy import misc
    import numpy as np

    def rgb2gray(rgb):
        """Convert RGB image to grayscale

          Parameters:
            rgb : RGB image

          Returns:
            gray : grayscale image

        """
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

    try:
        makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    num_frame = nib.load(smoothed_img).get_data().shape[-1]
    med2img_path = join(root_path, 'utils/med2image/med2image.py')

    if len(listdir(output_dir)) == 0:
        print "Either 2D png has not been converted or you forced to reslice it, use med2image to slice the merged image to 2D png images!!!"

        for i in xrange(num_frame):
            print 'Slice this frame into png: %s' % (str(i))
            cmd = "python " + med2img_path + " --inputFile " + smoothed_img + " --outputDir " + output_dir + " --outputFileStem " + output_file_stem + \
                  " --outputFileType " + output_file_type + " --frameToConvert " + str(i)
            system(cmd)

        ### convert the sliced png image into gray matter image
        for i in xrange(num_frame):
            for j in xrange(z_dimensions):
                print 'To convert png to grayscale image, the png is: %s' % ('c1-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(j) + '.png')
                png = join(output_dir, 'c1-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(j) + '.png')
                if png == join(output_dir, 'c1-frame690-slice090.png'):
                    continue
                else:
                    img = imread(
                        join(output_dir, 'c1-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(j) + '.png'))
                    grayimg = rgb2gray(img)
                    misc.imsave(join(output_dir, 'c1-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(j) + '-gray.png'), grayimg)

        ## delete the gray matter image whose value is always 0
        for i in xrange(num_frame):
            for j in xrange(z_dimensions):
                png = join(output_dir, 'c1-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(j) + '-gray.png')
                if png == join(output_dir, 'c1-frame690-slice090-gray.png'):
                    continue
                else:
                    img = imread(
                        join(output_dir, 'c1-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(j) + '-gray.png'))
                    if np.count_nonzero(img) == 0:
                        print 'Png always has 0 value, the png is: %s' % (
                        'c1-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(j) + '-gray.png')
                        remove(join(output_dir, 'c1-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(j) + '-gray.png'))

    else:
        print "Note: 2D png images exist, skip this step!!!"
    return output_dir


def nii_png_nibabel_scipy(smoothed_img, output_dir, output_file_stem, subject_list):
    """
    Convert a concatenated 4D nifti image to a 2D grayscale png image with nibabel and scipy package:
    using med2img to concert the 4D nifti into a rgb image; then convert them into a grayscale image, and delete the empty-intensity image
    :param smoothed_img: the concatenated smoothed 4D nifti
    :param output_dir:
    :param output_file_stem:
    :param subject_list: this is the lsit with the same order of smoothd_img
    :return:
    """
    from os import listdir, makedirs, remove
    from os.path import join, exists
    import nibabel as nib
    from scipy.misc import imsave, imread
    import numpy as np
    import errno
    import h5py

    try:
        makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    nii_data = nib.load(smoothed_img).get_data()
    num_frame = nii_data.shape[-1]
    cropped_axis_slices = nii_data.shape[0]

    ## check the dimension of the nifti
    print 'The shape of the nifit is: depth: %d, width: %d, length: %d !' % (nii_data.shape[0], nii_data.shape[1], nii_data.shape[2])

    if len(listdir(output_dir)) == 0:
        print "Either 2D png has not been converted or you forced to reslice it, use nibable to read the nii and scipy to store each slice as png!!!"

        for i in xrange(num_frame):
            subject = subject_list[i]
            print 'Slice this frame into png: %s th niift from the tsv files' % (str(i))
            frame_data = nii_data[:, :, :, i]
            j = 0
            for img in frame_data:
                print "Crop the nifti by the first dimension (depth in definition), the png image dimension: width: %d, length: %d" % (img.shape[0], img.shape[1])
                print "This image value range is from %f to %f" % (np.min(img), np.max(img))
                print "This png has %d non-zero voxels" % np.count_nonzero(img)
                if np.count_nonzero(img) <= 10000:
                    print 'Png always has less than certain non-zero values, the png is: %s' % (
                        subject + '-' + output_file_stem + '-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(
                            j) + '.png')
                else:
                    ## to save the array into png just for visualiztion
                    imsave(join(output_dir, subject + '-' + output_file_stem + '-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(j) + '.png'), img)
                    ## save the numpy array into hdf5
                    h5f = h5py.File(join(output_dir, subject + '-' + output_file_stem + '-frame' + '{:03}'.format(i) + '-slice' + '{:03}'.format(j) + '.h5'), 'w')
                    h5f.create_dataset("image", data=img)
                    h5f.close()

                j += 1

    else:
        print "Note: 2D png images exist, skip this step!!!"
    return output_dir

def get_info_from_tsv(output_dir, diagnosis_tsv):

    import pandas as pd

    subjects_visits = pd.io.parsers.read_csv(diagnosis_tsv, sep='\t')
    if (list(subjects_visits.columns.values)[0] != 'participant_id') and (list(subjects_visits.columns.values)[1] != 'session_id') and (list(subjects_visits.columns.values)[2] != 'diagnosis'):
        raise Exception('Subjects and visits file is not in the correct format.')
    subject_list = list(subjects_visits.participant_id)
    session_list = list(subjects_visits.session_id)
    subject_id = list(subject_list[i] + '_' + session_list[i] for i in range(len(subject_list)))
    subject_dir = []
    for i in range(len(subject_list)):
        subject = output_dir + '/' + subject_list[i] + '/' + session_list[i] + '/' + 't1' + '/' + 'freesurfer-cross-sectional'
        subject_dir.append(subject)
    return subject_dir, subject_id, subject_list, session_list
