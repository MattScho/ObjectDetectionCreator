'''
A nice program to build an object detector for arbitrary objects

See lines 244-246 to configure for your system

Created by combining aspects from the tutorial:
https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/


Authors: Matthew Schofield
Version: 4/22/2020
'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import pandas as pd
import xml.etree.ElementTree as ET

import io

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


import functools
import json
import os
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework

from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import trainer
from object_detection.utils import config_util

tf.logging.set_verbosity(tf.logging.INFO)


'''
Function step 2 of converting xml to csv
'''
def xml_to_csv(path):
    xml_list = []
    print(path)
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    names = []
    for i in xml_df['filename']:
        names.append(i + '.jpg')
    xml_df['filename'] = names
    return xml_df

'''
Function step 1 of converting xml to csvs

:param: projectPath - path to top level of program
'''
def convert_XML_to_CSV(projectPath):
    for d in ['trainImages', 'testImages', 'validationImages']:
        image_path = projectPath + 'data/Images/' + d
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv((projectPath + 'data/xml_labels/' + d + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')




def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, classMap):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(str(row['class']).encode('utf8'))
        classes.append(classMap[str(row['class'])])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def recordMaker(projectPath, classMap):
    csv_inputs = []
    image_dirs = []
    output_files = []

    # Objects to use
    '''
    objects = [str(i) for i in range(0,10)]
    objects.extend(["+", "-"])
    '''
    objects = ["trainImages", "testImages", "validationImages"]
    # generate files
    for obj in objects:
        csv_inputs.append(obj + "_labels.csv")
        image_dirs.append(obj)
        output_files.append(obj + ".record")

    for csv_input, image_dir, outputFile in zip(csv_inputs, image_dirs, output_files):
        writer = tf.python_io.TFRecordWriter(projectPath + "/data/records/" + outputFile)
        path = os.path.join(projectPath + "/data/Images/" + image_dir)
        examples = pd.read_csv(projectPath + "/data/xml_labels/" + csv_input)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path, classMap)
            writer.write(tf_example.SerializeToString())

        writer.close()
        output_path = os.path.join(projectPath + "/data/records/" + outputFile)
        print('Successfully created the TFRecords: {}'.format(output_path))

def trainModel(train_dir, pipelineFile):
    if True:
        configs = config_util.get_configs_from_pipeline_file(
            pipelineFile)
        if True:
            tf.gfile.Copy(pipelineFile,
                          os.path.join(train_dir, 'pipeline.config'),
                          overwrite=True)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    def get_next(config):
        return dataset_builder.make_initializable_iterator(
            dataset_builder.build(config)).get_next()

    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
        # Number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1
    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
        # Set up distributed training.
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                 job_name=task_info.type,
                                 task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=True)

    trainer.train(
        create_input_dict_fn,
        model_fn,
        train_config,
        master,
        task,
        1,
        worker_replicas,
        False,
        ps_tasks,
        worker_job_name,
        is_chief,
        train_dir,
        graph_hook_fn=graph_rewriter_fn)

PROJ_PATH = "D:/MajorProjects/ObjectDetectionCreator/"
PIPELINE_CONFIG = PROJ_PATH + "/data/faster_rcnn_inception_v2_coco.config"
TRAIN_DIR = PROJ_PATH + "/data/records/"

classMap = {
    "0": 10,
    "+": 11,
    "-": 12,
    "Loop 2": 13,
    "x": 14,
    "/": 15,
    ")": 16,
    "(": 17
}
for i in range(1, 10):
    classMap[str(i)] = i

convert_XML_to_CSV(PROJ_PATH)
recordMaker(PROJ_PATH, classMap)
trainModel(TRAIN_DIR, PIPELINE_CONFIG)