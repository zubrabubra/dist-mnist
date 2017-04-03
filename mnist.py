import tensorflow as tf
import math
import time
import os
import argparse
import sys
import ast

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None

num_classes = 10
image_size = 28
learning_rate = 0.01


def main(_):
    cluster_def = None

    CLUSTER_CONFIG = os.environ.get('CLUSTER_CONFIG')
    if CLUSTER_CONFIG:
        cluster_def = ast.literal_eval(CLUSTER_CONFIG)
    else:
        parameter_servers = ["ps:2222"]
        workers = ["worker0:2223", "worker1:2224", "worker2:2225"]
        cluster_def = {"ps": parameter_servers, "worker": workers}

    cluster = tf.train.ClusterSpec(cluster_def)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        with tf.device(tf.train.replica_device_setter(cluster=cluster)):
            images = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
            labels = tf.placeholder(tf.int32, shape=[None])

            with tf.name_scope('hidden1'):
                weights = tf.Variable(tf.truncated_normal([image_size * image_size, FLAGS.hidden1],
                                                          stddev=1.0 / math.sqrt(float(image_size * image_size))),
                                      name='weights')
                biases = tf.Variable(tf.zeros([FLAGS.hidden1]), name='biases')
                hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

            with tf.name_scope('hidden2'):
                weights = tf.Variable(tf.truncated_normal([FLAGS.hidden1, FLAGS.hidden2],
                                                          stddev=1.0 / math.sqrt(float(FLAGS.hidden1))),
                                      name='weights')
                biases = tf.Variable(tf.zeros([FLAGS.hidden2]), name='biases')
                hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

            with tf.name_scope('softmax_linear'):
                weights = tf.Variable(tf.truncated_normal([FLAGS.hidden2, num_classes],
                                                          stddev=1.0 / math.sqrt(float(FLAGS.hidden2))),
                                      name='weights')
                biases = tf.Variable(tf.zeros([num_classes]), name='biases')
                logits = tf.matmul(hidden2, weights) + biases

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int64(labels), logits=logits,
                                                                           name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='loss')

            tf.summary.scalar('loss', loss)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)

            correct = tf.nn.in_top_k(logits, labels, 1)
            eval = tf.reduce_sum(tf.cast(correct, tf.int32))

            summary = tf.summary.merge_all()
            init = tf.global_variables_initializer()

            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), global_step=global_step, init_op=init)

            with sv.prepare_or_wait_for_session(server.target) as sess:
                summary_writer = tf.summary.FileWriter(FLAGS.log_dir, tf.get_default_graph())

                for epoch in range(FLAGS.epochs):
                    batch_count = int(data_sets.train.num_examples / FLAGS.batch_size)

                    for batch in range(batch_count):
                        images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
                        feed_dict = {
                            images: images_feed,
                            labels: labels_feed
                        }
                        _, loss_val, sum_val, step_val = sess.run([train_op, loss, summary, global_step], feed_dict=feed_dict)

                        summary_writer.add_summary(sum_val, step_val)

                        if step_val % 1000 == 0:
                            print("Global Step: %d," % step_val,
                                  " Epoch: %2d," % epoch,
                                  " Step: %d," % batch,
                                  " Cost: %.4f" % loss_val)

                test_correct = sess.run(eval, feed_dict={images: data_sets.test.images, labels: data_sets.test.labels})
                print("Test accuracy: %2.2f" % (float(test_correct) / len(data_sets.test.images)))

                val_correct = sess.run(eval, feed_dict={images: data_sets.validation.images, labels: data_sets.validation.labels})
                print("Validation accuracy: %2.2f" % (float(val_correct) / len(data_sets.validation.images)))

            sv.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, default='/tmp/tensorflow/mnist/input_data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/fully_connected_feed')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden1', type=int, default=128)
    parser.add_argument('--hidden2', type=int, default=32)
    parser.add_argument('--fake_data', default=False, action='store_true')

    parser.add_argument('--job_name', type=str, default='')
    parser.add_argument('--task_index', type=int, default=0)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)