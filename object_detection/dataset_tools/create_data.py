##Custom Data_tool use for quiz-w8-data
import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')

FLAGS = flags.FLAGS

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
    img_path = os.path.join(image_subdirectory, data['filename'])
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    difficult_obj = []
    truncated = []
    poses = []

    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue
            difficult_obj.append(int(difficult))

            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])
            xmins.append(xmin / width)
            ymins.append(ymin / height)
            xmaxs.append(xmax / width)
            ymaxs.append(ymax / height)
            class_name = obj['name']
            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[class_name])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
    writer = tf.python_io.TFRecordWriter(output_filename)
    print(len(examples))
    for idx, example in enumerate(examples):
        if idx % 10 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')
        if not os.path.exists(xml_path):
            logging.warning('Could not find %s, ignoring example.', xml_path)
            continue
        with tf.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        try:
            tf_example = dict_to_tf_example(
                data,
                label_map_dict,
                image_dir)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', xml_path)

    writer.close()

def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    logging.info('Reading from Pet dataset.')
    image_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')
    # examples_path = os.path.join(annotations_dir, 'trainval.txt')
    # examples_list = dataset_util.read_examples_list(examples_path)
    #提供的数据集并没有trainval.txt，观测到examples_list就是读取的xml和图片的名称
    xmls_dir = os.path.join(annotations_dir, 'xmls')
    examples_list = []
    for file in os.listdir(xmls_dir):
        if os.path.splitext(file)[1] == '.xml':
            examples_list.append(os.path.splitext(file)[0])

    random.seed(3)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))
    train_output_path = os.path.join(FLAGS.output_dir, 'pet_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'pet_val.record')
    create_tf_record(
        train_output_path,
        label_map_dict,
        annotations_dir,
        image_dir,
        train_examples)
    create_tf_record(
        val_output_path,
        label_map_dict,
        annotations_dir,
        image_dir,
        val_examples)


if __name__ == '__main__':
  tf.app.run()