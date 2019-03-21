#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Junhao WEN"
__copyright__ = "Copyright 2016, The Aramis Lab Team"
__credits__ = ["Junhao WEN"]
__license__ = "??"
__version__ = "1.0.0"
__maintainer__ = "Junhao WEN"
__email__ = "junhao.wen@inria.fr"
__status__ = "Development"

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
from nipype.interfaces.utility import Function
from preprocessing_img_utils import *
from nipype.interfaces.fsl.utils import Merge, Smooth
from tempfile import mkdtemp
from os.path import join, realpath, split
import os

def preprocessing_img(caps_directory, output_dir, diagnosis_tsv, smooth_fwhm, working_directory=None, tissue='c1', modality='T1'):
    """
        Run preprocessing steps to cut 3D T1 images after Clinica Spm with a specific size which can be fit into your CNN model.
    Including the steps below:
        1. Get the preprocessed data by Clinica SPM from Jorge, the GM concatenated images(4D) for all the subjects for three groups(AD, CN, MCInc)
        2. Smooth the result GM images with nilearn (nilearn.image.smooth_img) or FSL.smooth with different smoothing kernel to explore if the classification accuracy would improve.
        3. Convert 4D concatenated GM images for three groups to 2D png format with a package med2image(https://github.com/FNNDSC/med2image) to check out the 0-pixel images of the GM images.
        4. Resize the 2D images to specific size according to your specific architectures.

    Note: need to think of the different dimension for all the images, if not the same dimension, should do something
    :return:
    :param caps_directory: the caps directory of T1-spm pipeline of Clinica
    :param output_dir: the image_preprocessing_output directory contain the sliced image
    :param diagnosis_tsv: the tsv contains the subjects that you want to process
    :param smooth_fwhm: the smooth
    :param working_directory:
    :param tissue: by default, c1==gray matter
    :return:
    """

    if working_directory is None:
        working_directory = mkdtemp()
    else:
        working_directory = os.path.abspath(working_directory)

    # Use datagrabber and inputnode to take the input images
    # get the info of 'subject_dir', 'subject_id', 'subject_list', 'session_list' from the tsv file.
    inputnode = pe.Node(name='inputnode',
                          interface=Function(
                          input_names=['output_dir', 'diagnosis_tsv'],
                          output_names=['subject_dir', 'subject_id', 'subject_list', 'session_list'],
                          function=get_info_from_tsv))
    inputnode.inputs.output_dir = output_dir
    inputnode.inputs.diagnosis_tsv = diagnosis_tsv

    # Node to grab the GM images of ADNI_baseline_t1 of SPM pipeline.
    datagrabbernode = pe.Node(interface=nio.DataGrabber(
                        infields=['subject_list', 'session_list', 'subject_repeat', 'session_repeat'],
                        outfields=['spm_tissuee']),
                        name="datagrabbernode") 
    datagrabbernode.inputs.template = '*'
    datagrabbernode.inputs.base_directory = caps_directory
    datagrabbernode.inputs.field_template = dict(spm_tissuee='subjects/%s/%s/t1/spm/dartel/group-ADNIbl/%s_%s_T1w_segm-graymatter_space-Ixi549Space_modulated-on_fwhm-' +
                                                             str(int(smooth_fwhm))+ 'mm_probability.nii.gz')
    datagrabbernode.inputs.template_args = dict(spm_tissuee=[['subject_list', 'session_list', 'subject_repeat',
                                                          'session_repeat']])
    datagrabbernode.inputs.sort_filelist = False

    ## Merge all the GM images to one big 4D image, the contatenated nifti is under output_dir/merged_nifti
    mergenode = pe.Node(name='mergeimgnibebel',
                        interface=Function(
                            input_names=['tissue_img', 'output_dir'],
                            output_names=['out_img_path'],
                            function=merge_img))
    mergenode.inputs.output_dir = output_dir

    # Slice 4D img to 2D jpg
    slicenode = pe.Node(name='nii2png_nibabel',
                        interface=Function(
                            input_names=[ 'smoothed_img', 'output_dir', 'output_file_stem', 'subject_list'],
                            output_names=['output_dir'],
                            function=nii_png_nibabel_scipy))
    slicenode.inputs.output_dir = join(output_dir, 'sliced_image')
    slicenode.inputs.output_file_stem = tissue

    ## Datasink node to grab every useful file that you need
    datasinknode = pe.Node(name='datasinker',
                           interface=nio.DataSink())
    datasinknode.inputs.base_directory = output_dir
    # datasinknode.inputs.container = 'prep_out'



    wf_preprocessing = pe.Workflow(name='preprocessing_nifti2png_2DCNN', base_dir=working_directory)

    wf_preprocessing.connect(inputnode, 'subject_list', datagrabbernode, 'subject_list')
    wf_preprocessing.connect(inputnode, 'session_list', datagrabbernode, 'session_list')
    wf_preprocessing.connect(inputnode, 'subject_list', datagrabbernode, 'subject_repeat')
    wf_preprocessing.connect(inputnode, 'session_list', datagrabbernode, 'session_repeat')
    wf_preprocessing.connect(datagrabbernode, 'spm_tissuee', mergenode, 'tissue_img')
    wf_preprocessing.connect(mergenode, 'out_img_path', slicenode, 'smoothed_img')
    wf_preprocessing.connect(inputnode, 'subject_list', slicenode, 'subject_list')
    wf_preprocessing.connect(mergenode, 'out_img_path', datasinknode, 'prep_out')


    return wf_preprocessing














