import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MODEL_DIR = "./checkpoint/"

def view_samples(samples, nrows, ncols, figsize=(5, 5)):

        fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharey=True, sharex=True)

        for ax, img in zip(axes.flatten(), samples):
            ax.axis('off')
            img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
            ax.set_adjustable('box-forced')
            im = ax.imshow(img, aspect='equal')
        plt.subplots_adjust(wspace=0, hspace=0)
        return fig, axes

g = tf.Graph()

with tf.Session(graph = g) as sess:
    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    g_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
    g_saver.restore(sess, ckpt.model_checkpoint_path)

    x_ops = g.get_operation_by_name("gen_input")
    y_ops = g.get_operation_by_name("generator/gen_output")

    sample_noise = np.random.uniform(low=-1, high=1, size=(72, 100))
    gen_samples = sess.run(y_ops.outputs[0], feed_dict={x_ops.outputs[0]: sample_noise})
    img_dir = "/home/tianweiz/image/image.png"
    fig, axes = view_samples(gen_samples, 6, 12, figsize=(10,5))
    fig.savefig(img_dir)
