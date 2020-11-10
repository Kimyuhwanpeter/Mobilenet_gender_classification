# -*- coding:utf-8 -*-
from absl import flags, app
from random import random, shuffle

import tensorflow as tf
import numpy as np
import os
import sys
import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

flags.DEFINE_integer("img_size", 224, "Image size")

flags.DEFINE_integer("batch_size", 16, "Training batch size")

flags.DEFINE_integer("epochs", 200, "Total training epochs")

flags.DEFINE_float("lr", 0.0001, "Learning rate")

flags.DEFINE_string("txt_path", "/yuhwan/yuhwan/Dataset/3rd_paper_dataset/[5]Gender_classification/Proposed_method/second_fold/AFAD/train.txt", "Training text path")

flags.DEFINE_string("img_path", "/yuhwan/yuhwan/Dataset/3rd_paper_dataset/[5]Gender_classification/Proposed_method/second_fold/AFAD/train/", "Training image path")

flags.DEFINE_string("test_txt_path", "/yuhwan/yuhwan/Dataset/3rd_paper_dataset/[5]Gender_classification/Proposed_method/second_fold/AFAD/test.txt", "Testing text path")

flags.DEFINE_string("test_img_path", "/yuhwan/yuhwan/Dataset/3rd_paper_dataset/[5]Gender_classification/Proposed_method/second_fold/AFAD/test/", "Testing image path")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("save_checkpoint", "/yuhwan/yuhwan/checkpoint/Gender_classification/Proposed_method/second_fold/AFAD/checkpoint", "Save checkpoint path")

flags.DEFINE_string("pre_checkpoint_path", "", "Restore the checkpoint path")

flags.DEFINE_string("graphs", "/yuhwan/yuhwan/checkpoint/Gender_classification/Proposed_method/second_fold/AFAD/graphs/", "Saving the training graphs")
# Bengali Ethnicity Recognition and Gender Classification Using CNN & Transfer Learning
FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def train_fuc(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size + 16, FLAGS.img_size + 16])
    img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3])
    #img = tf.image.per_image_standardization(img)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
    else:
        img = img

    img = tf.image.per_image_standardization(img)

    lab = tf.one_hot(lab_list, 2)
    # lab = lab_list

    return img, lab

def test_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    lab = lab_list

    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_race_loss(model, images, labels):

    with tf.GradientTape() as tape:
        logits = run_model(model, images, True)
        # logits = tf.nn.sigmoid(logits)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def cal_acc(model, images, labels):

    logits = run_model(model, images, False)
    logits = tf.nn.sigmoid(logits)  # [batch, 2]
    logits = tf.cast(tf.argmax(logits, 1), tf.float32)
    # logits = tf.squeeze(logits, 1)

    # predict = tf.cast(tf.greater(logits, 0.5), tf.float32)
    count_acc = tf.cast(tf.equal(logits, labels), tf.float32)
    count_acc = tf.reduce_sum(count_acc)

    return count_acc

def main(argv=None):
    # model = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
    #                    input_tensor=None, input_shape=(FLAGS.img_size, FLAGS.img_size, 3), pooling=None)
    model = tf.keras.applications.MobileNet(include_top=False, weights='imagenet',
                       input_tensor=None, input_shape=(FLAGS.img_size, FLAGS.img_size, 3), pooling="avg")
    regularizer = tf.keras.regularizers.l2(0.00001) # 다시 학습하자!
    
    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    x = model.output
    h = tf.keras.layers.Flatten()(x)
    h = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001))(h)
    h = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001))(h)
    y = tf.keras.layers.Dense(2, name='last_layer')(h)

    model = tf.keras.Model(inputs=model.input, outputs=y)
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("=====================================")
            print("* Restored the latest checkpoint!!! *")
            print("=====================================")

    if FLAGS.train:
        count = 0
        train_img = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
        train_img = [FLAGS.img_path + img for img in train_img]
        train_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        test_img = np.loadtxt(FLAGS.test_txt_path, dtype="<U100", skiprows=0, usecols=0)
        test_img = [FLAGS.test_img_path + img for img in test_img]
        test_lab = np.loadtxt(FLAGS.test_txt_path, dtype=np.float32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((test_img, test_lab))
        te_gener = te_gener.map(test_func)
        te_gener = te_gener.batch(FLAGS.batch_size)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        test_idx = len(test_img) // FLAGS.batch_size
        test_iter = iter(te_gener)
        
        #############################
        # Define the graphs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        train_log_dir = FLAGS.graphs + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = FLAGS.graphs + current_time + '/val'
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        #############################
        
        for epoch in range(FLAGS.epochs):
            
            tr = list(zip(train_img, train_lab))
            shuffle(tr)
            train_img, train_lab = zip(*tr)
            train_img, train_lab = np.array(train_img), np.array(train_lab)

            tr_gener = tf.data.Dataset.from_tensor_slices((train_img, train_lab))
            tr_gener = tr_gener.shuffle(len(train_img))
            tr_gener = tr_gener.map(train_fuc)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_idx = len(train_img) // FLAGS.batch_size
            train_iter = iter(tr_gener)

            for step in range(train_idx):
                batch_images, batch_labels = next(train_iter)

                loss = cal_race_loss(model, batch_images, batch_labels)
                
                with train_summary_writer.as_default():
                    tf.summary.scalar('Loss', loss, step=count)
                    
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step + 1, train_idx, loss))

                if count % 100 == 0:
                    test_idx = len(test_img) // FLAGS.batch_size
                    test_iter = iter(te_gener)
                    acc = 0.
                    for i in range(test_idx):
                        te_img, te_label = next(test_iter)

                        acc += cal_acc(model, te_img, te_label)

                    print("====================================")
                    print("step = {}, acc = {} %".format(count, (acc / len(test_img)) * 100.))
                    print("====================================")

                    with val_summary_writer.as_default():
                        tf.summary.scalar('Acc', (acc / len(test_img)) * 100., step=count)

                if count % 1000 == 0:
                    num_ = int(count) // 1000
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                    if not os.path.isdir(model_dir):
                        os.makedirs(model_dir)
                        print("Make {} files to save weight files (ckpt)")

                    ckpt = tf.train.Checkpoint(model=model, optim=optim)
                    ckpt_dir = model_dir + "/" + "bengail_ethnicy_gender_model_{}.ckpt".format(count)
                    ckpt.save(ckpt_dir)

                count += 1

if __name__ == "__main__":
    app.run(main)