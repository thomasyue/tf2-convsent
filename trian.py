import tensorflow as tf
import os
import logging
import joblib
import pickle
from models.convsent import ConvSent
from config import *
from models.utils import *
from tqdm import tqdm
import datetime
import traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
tf.random.set_seed(123)
np.random.seed(123)


@tf.function
def train_step(x, y, training=None):

    with tf.GradientTape() as tape:
        pred = model(x, y, training)
        losses = compute_loss(y, pred)

    grads = tape.gradient(losses, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return losses


def val_step(x, y, training=None):

    pred = model(x, y, training)
    losses = compute_loss(y, pred)

    return losses


def compute_loss(labels, predictions):
    mask = tf.math.logical_not(tf.math.equal(labels, 0))
    loss_object = tf.nn.sparse_softmax_cross_entropy_with_logits
    loss_ = loss_object(labels, predictions)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def main():
    total_loss = 0.0
    num_batches = 0
    total_steps = len(train) // batch_size
    total_val_steps = len(val) // batch_size
    try:
        for epoch in range(epochs):
            kf = get_minibatches_idx(len(train), batch_size, shuffle=True)
            val_loss = 0
            val_batches = 0
            for step, (_, train_index) in tqdm(enumerate(kf), total=total_steps, dynamic_ncols=True):
                sents = [train[t] for t in train_index]
                x = prepare_data_for_cnn(sents)
                y = np.array(prepare_data_for_rnn(sents))
                loss = train_step(x, y, training=True)
                total_loss += loss
                num_batches += 1
                current_loss = total_loss / num_batches
                global_train_step = epoch * total_steps + step

                del sents, x, y
                if step % 500 == 0:
                    template = "Epoch {}, Step {}, Current Loss: {:.5f}"
                    logger.info(template.format(epoch + 1, step, current_loss))
                    with train_summary_writer.as_default():
                        tf.summary.scalar('avg_loss', current_loss, step=global_train_step)

                if step % (total_steps//10) == 0:
                    save_path = manager.save()
                    template = "Epoch {}: Saved best ckpt to {}."
                    logger.info(template.format(epoch + 1, save_path))

            val_kf = get_minibatches_idx(len(val), batch_size)
            for step, (_, val_index) in tqdm(enumerate(val_kf), total=total_val_steps, dynamic_ncols=True):
                sents = [val[t] for t in val_index]
                x = prepare_data_for_cnn(sents)
                y = np.array(prepare_data_for_rnn(sents))
                loss = val_step(x, y, training=False)
                val_loss += loss
                val_batches += 1
                total_val_loss = val_loss / val_batches
                global_val_steps = epoch * total_val_steps + step

                del sents, x, y
                if step == total_val_steps-1:
                    with test_summary_writer.as_default():
                        tf.summary.scalar('val_loss', total_val_loss, step=global_val_steps)
                    template = "Epoch {}, Val Loss: {:.5f}"
                    logger.info(template.format(epoch + 1, total_val_loss))

    except Exception as ex:
        logger.info('Exception {}'.format(ex))
        logger.info('Traceback {}'.format(traceback.print_exc()))


if __name__ == '__main__':
    logger = logging.getLogger('train_autoencoder')
    logger.setLevel(logging.INFO)
    if word2vec:
        fh = logging.FileHandler('train_autoencoder_w2v.log')
    else:
        fh = logging.FileHandler('train_autoencoder_4w.log')
        # fh = logging.FileHandler('train_autoencoder_1M.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)

    # file_name = "../convsent/data/bookcorpus_1M.p"
    file_name = "./data/bookcorpus_4w.p"
    print('loading dataset from {}'.format(file_name))
    x = pickle.load(open(file_name, "rb"))
    train, val, test = x[0], x[1], x[2]

    wordtoix, ixtoword = x[6], x[7]
    del x

    n_words = len(ixtoword)
    ixtoword[n_words] = '<pad_zero>'
    wordtoix['<pad_zero>'] = n_words
    n_words = n_words + 1
    # del wordtoix, ixtoword

    if word2vec:
        checkpoint_dir += '/w2v_logs'
        w2v = joblib.load('../../pre_trained/happify_word2vec_with_stopwords.pkl')
        all_words = [x for x in w2v.wv.vocab.keys()]
        w2v_dict = {}

        for word in tqdm(all_words):
            w2v_dict[word] = w2v[word]

        embed_matrix = np.zeros((n_words, embed_dim))
        for word, idx in wordtoix.items():
            embed_vector = w2v_dict.get(word)
            if embed_vector is not None:
                embed_matrix[idx] = embed_vector
    else:
        embed_matrix = None

    model = ConvSent(
        n_words, embed_dim, hidden_dim, dropout_rate, feature_maps, filter_hs,
        embed_matrix
    )

    optimizer = tf.optimizers.Adam(learning_rate=lr)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    # ckpt_path = './logs/ckpt-4'
    # checkpoint.restore(ckpt_path)
    # logger.info("restored model from {}".format(ckpt_path))

    current_time = datetime.datetime.now().strftime("%m%d-%H%M")
    train_log_dir = './logs/tf_board/' + current_time + '/train'
    test_log_dir = './logs/tf_board/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    main()

