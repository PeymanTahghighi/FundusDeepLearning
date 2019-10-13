from Models import *
import tensorflow as tf
import Config
import pickle

placeholders = {
    'imageInput' : tf.placeholder(tf.float32, shape=(224, 224, 3),name="inputImage"),
    'labels': tf.placeholder(dtype=tf.float32, shape=(Config.NETWORK_SIZE * Config.NETWORK_SIZE, 1),name="Grountruth"),
    'features': tf.placeholder(tf.float32, shape=(None, 2),name="vertexCoordinates"),
}

sess = tf.Session();
sess.run(tf.global_variables_initializer());

# Build graph adjacency matrix
indices = [];
values = [];
for i in range(Config.NETWORK_SIZE * Config.NETWORK_SIZE):
    # self loop
    indices.append([i,i]);
    values.append(1.0);

    # Left edge vertices
    if i % Config.NETWORK_SIZE == 0:
        indices.append([i,i + 1]);
        values.append(1.0);
        if i - 1 > 0:
            indices.append([i,i - Config.NETWORK_SIZE]);
            values.append(1.0);
            # Top right
            indices.append([i,i - Config.NETWORK_SIZE + 1]);
            values.append(1.0);
        if i + 1 < Config.NETWORK_SIZE:
            indices.append([i,i + Config.NETWORK_SIZE]);
            values.append(1.0);
            # Bottom right
            indices.append([i,i + Config.NETWORK_SIZE + 1]);
            values.append(1.0);
    # --------------------------------------------------------

    # Right edge vertices
    elif i % (Config.NETWORK_SIZE - 1) == 0:
        indices.append([i,i - 1]);
        values.append(1.0);
        if i - 1 > 0:
            indices.append([i,i - Config.NETWORK_SIZE]);
            values.append(1.0);
            # Top left
            indices.append([i,i - Config.NETWORK_SIZE - 1]);
            values.append(1.0);
        if i + 1 < Config.NETWORK_SIZE:
            indices.append([i,i + Config.NETWORK_SIZE]);
            values.append(1.0);
            # Bottom left
            indices.append([i,i + Config.NETWORK_SIZE - 1]);
            values.append(1.0);
    # --------------------------------------------------------

    # Top row vertices
    elif 1 < i < Config.NETWORK_SIZE - 1:
        indices.append([i,i - 1]);
        values.append(1.0);
        indices.append([i,i + 1]);
        values.append(1.0);
        indices.append([i,i + Config.NETWORK_SIZE + 1]);
        values.append(1.0);
        indices.append([i,i + Config.NETWORK_SIZE - 1]);
        values.append(1.0);
        indices.append([i,i + Config.NETWORK_SIZE]);
        values.append(1.0);
    # --------------------------------------------------------

    # Bottom row vertices
    elif Config.NETWORK_SIZE * (Config.NETWORK_SIZE - 1) < i < Config.NETWORK_SIZE * Config.NETWORK_SIZE:
        indices.append([i,i - 1]);
        values.append(1.0);
        indices.append([i,i + 1]);
        values.append(1.0);
        indices.append([i,i - Config.NETWORK_SIZE + 1]);
        values.append(1.0);
        indices.append([i,i - Config.NETWORK_SIZE - 1]);
        values.append(1.0);
        indices.append([i,i - Config.NETWORK_SIZE]);
        values.append(1.0);
    # ----------------------------------------------------------

    # Other vertices
    else:
        indices.append([i,i - 1]);
        values.append(1.0);
        indices.append([i,i + 1]);
        values.append(1.0);
        indices.append([i,i - Config.NETWORK_SIZE]);
        values.append(1.0);
        indices.append([i,i + Config.NETWORK_SIZE]);
        values.append(1.0);

        indices.append([i,i - Config.NETWORK_SIZE + 1]);
        values.append(1.0);
        indices.append([i,i - Config.NETWORK_SIZE - 1]);
        values.append(1.0);
        if i == (Config.NETWORK_SIZE * (Config.NETWORK_SIZE - 1) - 1):
            indices.append([i,i + Config.NETWORK_SIZE]);
            values.append(1.0);
        else:
            indices.append([i,i + Config.NETWORK_SIZE + 1]);
            values.append(1.0);

        indices.append([i,i + Config.NETWORK_SIZE - 1]);
        values.append(1.0);

# ------------------------------------------------------------------------------------------

placeholders['graphNetwork'] = tf.SparseTensor(indices=indices,values=values,
                dense_shape=(Config.NETWORK_SIZE * Config.NETWORK_SIZE,Config.NETWORK_SIZE * Config.NETWORK_SIZE));

model = GCN(placeholders=placeholders);

coords = pickle.load(file=open("surface.dat","rb"));
feed_dict = dict();
feed_dict.update({placeholders['features']: coords})


init = tf.global_variables_initializer();
sess.run(init);

# Train model
for epoch in range(Config.EPOCHS):
    writer = tf.summary.FileWriter(logdir="./graphs",graph=sess.graph);
    img = load_img(path = "1.PNG");
    y_train = np.random.random_sample((Config.NETWORK_SIZE * Config.NETWORK_SIZE,1));

    feed_dict.update({placeholders['imageInput']: img})
    feed_dict.update({placeholders['labels']: y_train})

    loss = sess.run(model.opt_op,feed_dict=feed_dict);

    print("loss : {}".format(loss));

# -----------------------------------------------------