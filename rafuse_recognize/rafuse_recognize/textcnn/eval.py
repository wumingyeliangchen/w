import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_input_helper as data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters

# Data Parameters #数据参数
tf.flags.DEFINE_string("valid_data_file", "./data/valid_data.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("w2v_file", "./data/word2vec.bin", "w2v_file path")

# Eval Parameters #评估参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters #其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



def load_data(w2v_model,max_document_length = 20):
    """Loads starter word-vectors and train/dev/test data.""" #加载起始字向量和训练/dev/测试数据。
    # Load the starter word vectors #加载起始词向量
    print("Loading data...")
    x_text, y_test = data_helpers.load_data_and_labels(FLAGS.valid_data_file)
    y_test = np.argmax(y_test, axis=1)

    if(max_document_length == 0) :
        max_document_length = max([len(x.split(" ")) for x in x_text])

    print ('max_document_length = ' , max_document_length)

    x = data_helpers.get_text_idx(x_text,w2v_model.vocab_hash,max_document_length)


    return x,y_test

def eval(w2v_model):
    # Evaluation #评价
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables #加载保存的元图并恢复变量
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name #从图表中按名称获取占位符
            input_x = graph.get_operation_by_name("input_x").outputs[0]
          
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate #我们要计算的张量
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            x_test, y_test = load_data(w2v_model, 5)
            # Generate batches for one epoch #为一个epoch生成批处理
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here #在这里收集预测
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined #如果定义了y检验，则打印精度
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Save the evaluation to a csv #将评估保存到csv
    predictions_human_readable = np.column_stack(all_predictions)
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)



if __name__ == "__main__":
    w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
    eval(w2v_wr.model)
