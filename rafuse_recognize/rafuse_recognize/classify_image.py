# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.
Run image classification with Inception trained on ImageNet 2012 Challenge data
set.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.
Change the --image_file argument to any jpg image to compute a
classification of that image.
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.
https://tensorflow.org/tutorials/image_recognition/
"""
#__future__这是为了在是py2的时候还能用到一些新版本的特性而做成的包 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
#urlib作用网页请求,响应获取,代理和cookie设置,异常处理,URL解析    https://www.jianshu.com/p/87d1e2f875b7
# import tensorflow as tf
import tensorflow.compat.v1 as tf
#FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self, 
                uid_chinese_lookup_path,  #uid_中文_查找_路径，根据前面的UID找到后面对应的文字
                model_dir, 
                label_lookup_path=None,
                uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(        #os.path.join()函数：连接两个或更多的路径名组件
          model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          model_dir, 'imagenet_synset_to_human_label_map.txt')
    #self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
    self.node_lookup = self.load_chinese_map(uid_chinese_lookup_path)
	
	

  def load(self, label_lookup_path, uid_lookup_path): #好像根本没用到这个函数
    """为每个softmax节点加载可读的英文名称。
    参数：
      label_lookup_path：字符串UID到整数节点ID。
      uid_lookup_path：字符串uid到人类可读字符串。
    退货：
     从整数节点ID到人类可读字符串的dict.
    """
    if not tf.gfile.Exists(uid_lookup_path): #判断目录或文件是否存在，filename可为目录路径或带文件名的路径，有该目录则返回True，否则False。
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    #p = re.compile(r'[n\d]*[ \S,]*')
    p = re.compile(r'(n\d*)\t(.*)')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      print(parsed_items)
      uid = parsed_items[0]
      human_string = parsed_items[1]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID. 
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name
    
  def load_chinese_map(self, uid_chinese_lookup_path):
    # Loads mapping from string UID to human-readable string #将UID从字符串加载到可读字符串
    proto_as_ascii_lines = tf.gfile.GFile(uid_chinese_lookup_path).readlines()
	#tf.gfile.GFile(filename, mode) 获取文本操作句柄，类似于python提供的文本操作open()函数，filename是要打开的文件名，mode是以何种方式去读写，将会返回一个文本操作句柄。
    uid_to_human = {}
    p = re.compile(r'(\d*)\t(.*)')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      #print(parsed_items)
      uid = parsed_items[0][0]
      human_string = parsed_items[0][1]
      uid_to_human[int(uid)] = human_string
    
    return uid_to_human

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph(model_dir): #使用Inception-v3模型，对图像进行处理
  """Creates a graph from saved GraphDef file and returns a saver.""" 
  # Creates graph from saved graph_def.pb. 
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f: 
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
	#这个函数提供了一种方法来导入序列化的TensorFlow GraphDef协议缓冲区，并将GraphDef中的各个对象提取为tf。
	#graph_def: 包含要导入到默认图中的操作的GraphDef proto。
	#name: (可选.) 将前缀放在graph_def中名称前面的前缀。注意，这并不适用于导入的函数名。默认为"import".

def run_inference_on_image(image): #根据处理的图片，推断出对应的物品名称
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image): #如果路径没找到图片，输出未找到文件日志信息
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph(FLAGS.model_dir)

  with tf.Session() as sess: 
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across 包含规范化预测的张量
    #   1000 labels. 1000个标签。
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 包含包含2048的倒数第二层的张量 
    #   float description of the image. 图像的浮点描述。
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG 包含提供JPEG的字符串的张量
    #   encoding of the image. 图像的编码。
    # Runs the softmax tensor by feeding the image_data as input to the graph. 通过将图像数据作为输入输入到graph形来运行softmax张量。
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> chinese string lookup. 创建节点标识-->中文字符串查找。
    node_lookup = NodeLookup(uid_chinese_lookup_path='./data/imagenet_2012_challenge_label_chinese_map.pbtxt', \
                                model_dir=FLAGS.model_dir)

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
      #print('node_id: %s' %(node_id))


def maybe_download_and_extract():
  """Download and extract model tar file.""" #下载并提取模型tar文件。
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):  #判断dest_directory变量的路径文件夹是否存在，不存在则新建
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]  
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath): #如果inception-2015-12-05.tgz压缩包没在文件夹找到，则在DATA_URL对应网址下载下来
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory) #解压文件到dest_directory变量的路径下


def main(_):
  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg')) #如果没输入图片路径，默认打开那张熊猫的图片
  run_inference_on_image(image)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()  #FLAGS是上面parser输入的参数
  #parse_known_args()方法的作用就是当仅获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)#tf.app.run的核心意思：执行程序中main函数，并解析命令行参数！
