import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def view_samples(samples, nrows, ncols, figsize=(5, 5)):

        fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharey=True, sharex=True)

        for ax, img in zip(axes.flatten(), samples):
            ax.axis('off')
            img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
            ax.set_adjustable('box-forced')
            im = ax.imshow(img, aspect='equal')
        plt.subplots_adjust(wspace=0, hspace=0)
        return fig, axes


def Convert2Image(array, file_location):
    array = array.reshape(32, 32, 3)
    data = ((array - array.min())*255 / (array.max() - array.min())).astype(np.uint8)
    img = Image.fromarray(data, 'RGB')
    img.save(file_location)


def generate_save_samples(model_dir, data_dir, num_samples, save_format = "raw"):

    config = tf.ConfigProto(device_count = {'GPU': 0})

    g = tf.Graph()
    with tf.Session(graph = g, config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        g_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
        g_saver.restore(sess, ckpt.model_checkpoint_path)

        x_ops = g.get_operation_by_name("gen_input")
        y_ops = g.get_operation_by_name("generator/gen_output")

        sample_noise = np.random.uniform(low=-1, high=1, size=(num_samples, 100))
        gen_samples = sess.run(y_ops.outputs[0], feed_dict={x_ops.outputs[0]: sample_noise})

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if save_format == "image":
            for i in range(num_samples):
               Convert2Image(gen_samples[i], data_dir + "/" + str(i) + ".png")

        elif save_format == "raw":
            gen_samples = (gen_samples - gen_samples.min())/(gen_samples.max() - gen_samples.min())
            gen_samples = gen_samples.transpose((0, 3, 1, 2)).reshape((-1, 3072))
            np.save(data_dir + "/raw_data.npy", gen_samples)


#        img_dir = "/home/tianweiz/image/image.png"
#        fig, axes = view_samples(gen_samples, 6, 12, figsize=(10,5))
#        fig.savefig(img_dir)

model_dir = "/home/tianweiz/saved_models/checkpoint/"
data_dir = "/home/tianweiz/DeepLearningPrivacy/GAN/data"
generate_save_samples(model_dir, data_dir, 1000)
