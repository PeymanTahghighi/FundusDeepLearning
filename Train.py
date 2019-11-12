from Models import *
import tensorflow as tf
import Config
import pickle
import Fetcher

batch_size = 10;

def addSparse(x,y,counter,indices,values):
    indices.append([x,y]);
    values.append(1.0);
    counter[0]+=1;

placeholders = {
    'imageInput' : tf.placeholder(tf.float32, shape=(batch_size,100, 100, 3),name="inputImage"),
    'labels': tf.placeholder(dtype=tf.float32, shape=(batch_size,Config.NETWORK_SIZE * Config.NETWORK_SIZE),name="Grountruth"),
}

sess = tf.Session();
sess.run(tf.global_variables_initializer());


# Build graph adjacency matrix
indices = [];
values = [];
indicesDeg = [];
valuesDeg = [];
powNetSize = pow(Config.NETWORK_SIZE,2);
for i in range(Config.NETWORK_SIZE * Config.NETWORK_SIZE):
    counter = [1];
    counter[0] =0;
    # self loop
    addSparse(x=i,y=i,counter=counter,indices=indices,values=values);

    # Left edge vertices
    if i % Config.NETWORK_SIZE == 0:
        addSparse(x=i, y=i+1, counter=counter, indices=indices, values=values);
        if i - 1 > 0:
            addSparse(x=i, y=i - Config.NETWORK_SIZE,
                      counter=counter, indices=indices, values=values);
            # Top right
            addSparse(x=i, y=i - Config.NETWORK_SIZE+1,
                      counter=counter, indices=indices, values=values);
        if i + Config.NETWORK_SIZE < powNetSize:
            addSparse(x=i, y=i + Config.NETWORK_SIZE,
                      counter=counter, indices=indices, values=values);
            # Bottom right
            addSparse(x=i, y=i + Config.NETWORK_SIZE + 1,
                      counter=counter, indices=indices, values=values);
    # --------------------------------------------------------

    # Right edge vertices
    elif (i+1) % (Config.NETWORK_SIZE) == 0:
        addSparse(x=i, y=i - 1,
                  counter=counter, indices=indices, values=values);
        if i - Config.NETWORK_SIZE > 0:
            addSparse(x=i, y=i - Config.NETWORK_SIZE,
                      counter=counter, indices=indices, values=values);
            # Top left
            addSparse(x=i, y=i - Config.NETWORK_SIZE - 1,
                      counter=counter, indices=indices, values=values);
        if i + Config.NETWORK_SIZE < powNetSize:
            addSparse(x=i, y=i + Config.NETWORK_SIZE,
                      counter=counter, indices=indices, values=values);
            # Bottom left
            addSparse(x=i, y=i + Config.NETWORK_SIZE-1,
                      counter=counter, indices=indices, values=values);

    # --------------------------------------------------------

    # Top row vertices
    elif 1 <= i <= Config.NETWORK_SIZE - 1:
        addSparse(x=i, y=i - 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i + 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i + Config.NETWORK_SIZE + 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i + Config.NETWORK_SIZE - 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i + Config.NETWORK_SIZE,
                  counter=counter, indices=indices, values=values);
    # --------------------------------------------------------

    # Bottom row vertices
    elif Config.NETWORK_SIZE * (Config.NETWORK_SIZE - 1) < i < Config.NETWORK_SIZE * Config.NETWORK_SIZE:
        addSparse(x=i, y=i - 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i + 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i - Config.NETWORK_SIZE + 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i - Config.NETWORK_SIZE - 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i - Config.NETWORK_SIZE,
                  counter=counter, indices=indices, values=values);
    # ----------------------------------------------------------

    # Other vertices
    else:
        addSparse(x=i, y=i - 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i + 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i - Config.NETWORK_SIZE,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i + Config.NETWORK_SIZE,
                  counter=counter, indices=indices, values=values);


        addSparse(x=i, y=i - Config.NETWORK_SIZE + 1,
                  counter=counter, indices=indices, values=values);
        addSparse(x=i, y=i - Config.NETWORK_SIZE - 1,
                  counter=counter, indices=indices, values=values);
        if i == (Config.NETWORK_SIZE * (Config.NETWORK_SIZE - 1) - 1):
            addSparse(x=i, y=i + Config.NETWORK_SIZE,
                      counter=counter, indices=indices, values=values);
        else:
            addSparse(x=i, y=i + Config.NETWORK_SIZE + 1,
                      counter=counter, indices=indices, values=values);

        addSparse(x=i, y=i + Config.NETWORK_SIZE - 1,
                  counter=counter, indices=indices, values=values);

    indicesDeg.append([i,i]);
    valuesDeg.append(1 / counter[0]);

# ------------------------------------------------------------------------------------------

placeholders['graphNetwork'] = tf.SparseTensor(indices=indices,values=values,
                dense_shape=(Config.NETWORK_SIZE * Config.NETWORK_SIZE,Config.NETWORK_SIZE * Config.NETWORK_SIZE));
placeholders['graphDegree'] = tf.SparseTensor(indices=indicesDeg,values=valuesDeg,
                dense_shape=(Config.NETWORK_SIZE * Config.NETWORK_SIZE,Config.NETWORK_SIZE * Config.NETWORK_SIZE));

coords = pickle.load(file=open("surface.dat","rb"));
placeholders['features'] = coords;
model = GCN(placeholders=placeholders);

#Utils.createCoords();

feed_dict = dict();


data = Fetcher.DataFetcher("Dataset\heightmap","Dataset\image");
data.load();

init = tf.global_variables_initializer();
sess.run(init);
writer = tf.summary.FileWriter(logdir="./graphs",graph=sess.graph);

num_iter = data.getTrainDataSize();
# Train model
for epoch in range(Config.EPOCHS):
    print("[INFO]Begin epoch {}...".format(epoch));
    all_loss = np.zeros(int((num_iter/batch_size)),dtype='float32');

    for iter in range(int((num_iter/batch_size))):

        img, y_train = data.fetchTrain(batchSize=batch_size);
        # img = load_img(path = "1.PNG");
        # y_train = np.random.random_sample((Config.NETWORK_SIZE * Config.NETWORK_SIZE,1));

        feed_dict.update({placeholders['imageInput']: img})
        feed_dict.update({placeholders['labels']: y_train})

        _,loss = sess.run([model.opt_op,model.loss],feed_dict=feed_dict);
        all_loss[iter] = loss;
        mean_loss = np.mean(all_loss[np.where(all_loss)])
        print('Epoch %d, Iteration %d' % (epoch + 1, iter + 1));
        print('Mean loss = %f, iter loss = %f' % (mean_loss, loss));

# -----------------------------------------------------