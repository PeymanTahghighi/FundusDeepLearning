import tensorflow as tf
import Fetcher
import numpy as np
from matplotlib import pyplot as plt

tf.reset_default_graph()
data = Fetcher.DataFetcher("Dataset\heightmap","Dataset\image");
data.load();

random_pred = 10;

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model.mdl-560.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    print("Model restored successfully...");
    graph = tf.get_default_graph();
    x = graph.get_tensor_by_name("Input/X:0");
    y = graph.get_tensor_by_name("Input/Y:0");
    op_pred = graph.get_tensor_by_name("Train/Prediction/predictions:0")

    for i in range(random_pred):
        img, rad = data.getRandomData();

        a = np.argmax(rad, 1);
        feed_dict = {x: img,y: rad};
        pred = sess.run(op_pred, feed_dict = feed_dict);
        plt.imshow(img.squeeze());
        plt.title("Prediction : " + str(pred) + "\n" + "True : " + str(np.argmax(rad,1)))
        plt.show();
        print("Prediction : " + str(pred));
        print("True : " + str(np.argmax(rad,1)));

#
#
#pred = sess.run('Train/Prediction/predictions:0',feed_dict);
#print(pred);