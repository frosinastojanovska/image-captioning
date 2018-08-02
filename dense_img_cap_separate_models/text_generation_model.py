import os
import json
import keras
import pickle
import numpy as np
import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import skimage.io as skiimage_io
import skimage.color as skimage_color
from keras.preprocessing.sequence import pad_sequences

from utils import Dataset
from config import Config
from modified_dense_model import BatchNorm
from generate_one_roi_features import generate_features, load_model
from preprocess import encode_caption, load_corpus, load_embeddings, tokenize_corpus, decode_caption

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class DenseCapConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "dense image captioning"

    # Train on 1 GPU and n images per GPU. Batch size is n (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BATCH_SIZE = 5

    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50

    # Padding size
    PADDING_SIZE = 10

    def __init__(self, vocab_size, embedding_weights):
        super(DenseCapConfig, self).__init__()
        # Vocabulary size
        self.VOCABULARY_SIZE = vocab_size
        self.EMBEDDING_WEIGHTS = embedding_weights
        self.EMBEDDING_SIZE = embedding_weights.shape[1]


class VisualGenomeDataset(Dataset):
    def __init__(self, words_to_ids, padding_size):
        super(VisualGenomeDataset, self).__init__()
        self.word_to_id = words_to_ids
        self.padding_size = padding_size

    def load_visual_genome(self, data_dir, image_ids, image_meta_file, data_file):
        with open(data_file, 'r', encoding='utf-8') as doc:
            img_data = json.loads(doc.read())
        data = dict()
        for x in img_data:
            data[x['id']] = x['regions']

        with open(image_meta_file, 'r', encoding='utf-8') as doc:
            img_meta = json.loads(doc.read())
        image_meta = dict()
        for x in img_meta:
            image_meta[x['image_id']] = x

        # Add images
        for i in image_ids:
            captions = [[d['phrase']] for d in data[i]]
            self.add_image(
                "VisualGenome", image_id=i,
                path=os.path.join(data_dir, '{}.jpg'.format(i)),
                width=image_meta[i]['width'],
                height=image_meta[i]['height'],
                rois=[[d['y'], d['x'], d['y'] + d['height'], d['x'] + d['width']] for d in data[i]],
                captions=captions
            )

    def image_reference(self, image_id):
        """ Return a link to the image in the Stanford Website. """
        info = self.image_info[image_id]
        return "https://cs.stanford.edu/people/rak248/VG_100K/{}.jpg".format(info["id"])

    def load_image(self, image_id):
        """ Load the specified image and return a [H,W,3] Numpy array. """
        # Load image
        image = skiimage_io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage_color.gray2rgb(image)
        return image

    def load_captions_and_rois(self, image_id):
        """ Generate instance captions of the given image ID. """
        image_info = self.image_info[image_id]
        rois = []
        caps = []
        for roi, caption in zip(image_info['rois'], image_info['captions']):
            cap = self.encode_region_caption(caption[0])
            if cap.size != 0:
                rois.append(roi)
                caps.append(cap)
        captions = pad_sequences(caps, maxlen=self.padding_size, padding='post', dtype='float').astype(np.float32)
        rois = np.array(rois)
        return rois, captions

    def load_original_captions_and_rois(self, image_id):
        """ Returns string instance captions of the given image ID. """
        image_info = self.image_info[image_id]
        rois = np.array(image_info['rois'])
        captions = image_info['captions']
        return rois, captions

    def encode_region_caption(self, caption):
        """ Convert caption to word embedding vector """
        return encode_caption(caption, self.word_to_id)


def build_roi_caption_model(mode, features_input, lstm_units, config):
    if mode == 'training':
        inputs = KL.Input(batch_shape=[config.BATCH_SIZE] + features_input, name="input_imgcap_caption_features")
        feature = KL.Lambda(lambda x: x[:, :, :-config.PADDING_SIZE], name="imgcap_caption_feature")(inputs)
        gt_captions = KL.Lambda(lambda x: x[:, :, -config.PADDING_SIZE:], name="imgcap_caption_gt_captions")(inputs)
    else:
        inputs = KL.Input(batch_shape=[config.BATCH_SIZE] + features_input, name="input_imgcap_caption_features")
        feature = KL.Lambda(lambda x: x, name="imgcap_caption_feature")(inputs)
    last_word = KL.Lambda(lambda x: tf.zeros([config.BATCH_SIZE, x.shape[1]], tf.float32),
                          output_shape=[feature.shape.as_list()[1]], name='imgcap_last_word')(feature)

    embedding = KL.Embedding(input_dim=config.VOCABULARY_SIZE,
                             output_dim=config.EMBEDDING_SIZE,
                             weights=[config.EMBEDDING_WEIGHTS],
                             trainable=False,
                             mask_zero=True,
                             name='imgcap_embedding_layer')
    lstm = KL.LSTM(units=lstm_units, recurrent_dropout=0.2,
                   return_state=False, stateful=False, name='imgcap_lstm')
    dense1 = KL.Dense(1024, activation='relu', name='imgcap_lstm_d1')
    dense2 = KL.Dense(config.VOCABULARY_SIZE, activation='softmax', name='imgcap_lstm_d2')

    concat = KL.Concatenate(axis=-1, name='img_cap_concat')

    img_cap_caption = []
    for j in range(config.PADDING_SIZE):
        if j == 0:
            embedd = KL.Lambda(lambda x: tf.zeros([config.BATCH_SIZE, 1, config.EMBEDDING_SIZE],
                                                  tf.float32))(last_word)
        else:
            embedd = embedding(last_word)
        concatenated_input = concat([feature, embedd])
        rnn = lstm(concatenated_input)
        feature_new = KL.Lambda(lambda x: tf.squeeze(x, axis=[1]))(feature)
        concatenated_output = concat([feature_new, rnn])
        out1 = dense1(concatenated_output)
        current_word_prob = dense2(out1)
        img_cap_caption.append(current_word_prob)
        if mode == 'training':
            last_word = KL.Lambda(lambda x: x[:, :, j])(gt_captions)
        else:
            last_word = KL.Lambda(lambda x: tf.cast(tf.expand_dims(tf.argmax(x, axis=-1), axis=1),
                                                    tf.float32))(current_word_prob)

    img_cap_caption = KL.Lambda(lambda x: tf.stack(x, axis=1))(img_cap_caption)

    return KM.Model(inputs, img_cap_caption, name="imgcap_caption_model")


def build_roi_caption_model_second_type(mode, features_input, lstm_units, config):
    if mode == 'training':
        inputs = KL.Input(batch_shape=[config.BATCH_SIZE] + features_input, name="input_imgcap_caption_features")
        feature = KL.Lambda(lambda x: x[:, :, :-config.PADDING_SIZE], name="imgcap_caption_feature")(inputs)
        gt_captions = KL.Lambda(lambda x: x[:, :, -config.PADDING_SIZE:], name="imgcap_caption_gt_captions")(inputs)
    else:
        inputs = KL.Input(batch_shape=[config.BATCH_SIZE] + features_input, name="input_imgcap_caption_features")
        feature = KL.Lambda(lambda x: x, name="imgcap_caption_feature")(inputs)
    current_caption = KL.Lambda(lambda x: tf.zeros([config.BATCH_SIZE, config.PADDING_SIZE], tf.float32),
                                output_shape=[config.PADDING_SIZE],
                                name='imgcap_last_word')(feature)

    embedding = KL.Embedding(input_dim=config.VOCABULARY_SIZE,
                             output_dim=config.EMBEDDING_SIZE,
                             weights=[config.EMBEDDING_WEIGHTS],
                             trainable=False,
                             mask_zero=True,
                             name='imgcap_embedding_layer')
    prev_lstm = KL.LSTM(units=lstm_units, name='imgcap_lstm1')
    lstm = KL.LSTM(units=lstm_units, recurrent_dropout=0.2,
                   return_state=False, stateful=False, name='imgcap_lstm2')
    dense1 = KL.Dense(1024, activation='relu', name='imgcap_lstm_d1')
    dense2 = KL.Dense(config.VOCABULARY_SIZE, activation='softmax', name='imgcap_lstm_d2')

    concat = KL.Concatenate(axis=-1, name='img_cap_concat')

    img_cap_caption = []
    for j in range(config.PADDING_SIZE):
        if j == 0:
            embedd = KL.Lambda(lambda x: tf.zeros([config.BATCH_SIZE, config.PADDING_SIZE, config.EMBEDDING_SIZE],
                                                  tf.float32))(current_caption)
        else:
            embedd = embedding(current_caption)
        rnn_words = prev_lstm(embedd)
        rnn_words = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(rnn_words)
        concatenated_input = concat([feature, rnn_words])
        rnn = lstm(concatenated_input)
        feature_new = KL.Lambda(lambda x: tf.squeeze(x, axis=[1]))(feature)
        concatenated_output = concat([feature_new, rnn])
        out1 = dense1(concatenated_output)
        current_word_prob = dense2(out1)
        img_cap_caption.append(current_word_prob)
        if mode == 'training':
            current_caption = KL.Lambda(lambda x: tf.pad(tf.squeeze(x[:, :, :(j+1)], axis=1),
                                                         tf.constant([[0, 0],
                                                                      [0, config.PADDING_SIZE - j - 1]])))(gt_captions)
        else:
            if len(img_cap_caption) > 1:
                current_caption = KL.Lambda(lambda x: tf.pad(tf.argmax(tf.stack(x, axis=1), axis=-1),
                                                             tf.constant([[0, 0],
                                                                          [0, config.PADDING_SIZE - j - 1]])
                                                             ))(img_cap_caption)
            else:
                current_caption = KL.Lambda(lambda x: tf.pad(tf.expand_dims(tf.argmax(x, axis=-1), axis=-1),
                                                             tf.constant([[0, 0],
                                                                          [0, config.PADDING_SIZE - j - 1]])
                                                             ))(img_cap_caption[0])

    img_cap_caption = KL.Lambda(lambda x: tf.stack(x, axis=1))(img_cap_caption)
    return KM.Model(inputs, img_cap_caption, name="imgcap_caption_model")


def build_lstm_model(features_input, config, units):
    """Builds a Keras model of the LSTM Caption Generation Network.
    It wraps the lstm generator graph so it can be used multiple times with shared
    weights.

    features_input: expected shape of the features as input (without batch size)
    config: Configuration object.
    units: Number of units of the LSTM network.

    Returns a Keras Model object. The model output, when called, is:
    generated_words: [batch, NUMBER OF ROIS, PADDING_SIZE, VOCABULARY_SIZE] Words of captions for each ROI.
    """
    features = KL.Input(batch_shape=[config.BATCH_SIZE] + features_input, name="input_imgcap_lstm_features")
    gt_captions = KL.Input(shape=[config.PADDING_SIZE],
                           batch_shape=[config.BATCH_SIZE, config.PADDING_SIZE],
                           name="input_imgcap_lstm_gt_captions")

    features_new = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(features)
    gt_captions_new = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(gt_captions)

    x = KL.TimeDistributed(KL.Conv2D(1024, (config.POOL_SIZE, config.POOL_SIZE), padding="valid", trainable=False),
                           name="mrcnn_class_conv1")(features_new)
    x = KL.TimeDistributed(BatchNorm(axis=3, trainable=False), name='mrcnn_class_bn1')(x)
    x = KL.Activation('relu', trainable=False)(x)
    x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1), trainable=False),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(axis=3, trainable=False),
                           name='mrcnn_class_bn2')(x)
    x = KL.Activation('relu', trainable=False)(x)

    features_new = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                             name="pool_squeeze")(x)

    merged_input = KL.Concatenate(axis=-1)([features_new, gt_captions_new])
    merged_input = KL.Lambda(lambda x: tf.expand_dims(x, axis=2))(merged_input)
    caption_model = build_roi_caption_model_second_type('training', merged_input.shape.as_list()[2:],
                                                        units, config)
    outputs = KL.TimeDistributed(caption_model)(merged_input)
    outputs = KL.Lambda(lambda x: tf.squeeze(x, axis=1))(outputs)
    return KM.Model([features, gt_captions], outputs, name="imgcap_lstm_model")


def build_train_data(dataset, model):
    if os.path.exists('train_x.pkl'):
        with open('train_x.pkl', 'rb') as f:
            train_x = pickle.load(f)
        with open('train_y.pkl', 'rb') as f:
            train_y = pickle.load(f)
    else:
        print('Building dataset ...')
        train_x = []
        train_y = []
        for image_id in dataset._image_ids:
            features = generate_features(dataset, image_id, model)
            _, captions = dataset.load_captions_and_rois(image_id)
            steps = features.shape[0]
            for i in range(steps):
                train_x.append(features[i])
                train_y.append(captions[i])
        with open('train_x.pkl', 'wb') as f:
            pickle.dump(train_x, f, pickle.HIGHEST_PROTOCOL)
        with open('train_y.pkl', 'wb') as f:
            pickle.dump(train_y, f, pickle.HIGHEST_PROTOCOL)

    return np.array(train_x), np.array(train_y)


if __name__ == '__main__':
    embeddings_file_path = '../dataset/glove.6B.300d.txt'

    data_directory = '../dataset/visual genome/'
    image_meta_file_path = '../dataset/image_data.json'
    data_file_path = '../dataset/region_descriptions.json'

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    train_image_ids = [1, 2, 55, 56, 62, 65, 77]

    # load word ids
    id_to_word_file = '../dataset/id_to_word.pickle'
    word_to_id_file = '../dataset/word_to_id.pickle'
    embedding_matrix_file = '../dataset/embedding_matrix.pickle'

    if not os.path.exists(id_to_word_file) or not os.path.exists(word_to_id_file) \
            or not os.path.exists(embedding_matrix_file):
        embeddings = load_embeddings(embeddings_file_path)
        tokens = tokenize_corpus(data_file_path, train_image_ids, embeddings)
        word_to_id, id_to_word, embedding_matrix = load_corpus(list(tokens), embeddings, 300)

        with open(id_to_word_file, 'wb') as f:
            pickle.dump(id_to_word, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(word_to_id_file, 'wb') as f:
            pickle.dump(word_to_id, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(embedding_matrix_file, 'wb') as f:
            pickle.dump(embedding_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(id_to_word_file, 'rb') as f:
            id_to_word = pickle.load(f)
        with open(word_to_id_file, 'rb') as f:
            word_to_id = pickle.load(f)
        with open(embedding_matrix_file, 'rb') as f:
            embedding_matrix = pickle.load(f)

    config = DenseCapConfig(len(id_to_word), embedding_matrix)
    config.display()

    # Training dataset
    dataset_train = VisualGenomeDataset(word_to_id, config.PADDING_SIZE)
    dataset_train.load_visual_genome(data_directory, train_image_ids,
                                     image_meta_file_path, data_file_path)
    dataset_train.prepare()

    features_model = load_model()

    train_X, train_y = build_train_data(dataset_train, features_model)
    # train_X, train_y = build_train_data(dataset_train, None)

    train_x = train_y.copy()

    train_y = keras.utils.to_categorical(train_y, num_classes=config.VOCABULARY_SIZE)

    model_filepath = 'models/model-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/text_generation.log'

    model = build_lstm_model(list(train_X[0].shape), config, 512)
    opt = keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=opt,
                  loss=keras.losses.categorical_crossentropy)
    model._make_train_function()
    checkpoint = keras.callbacks.ModelCheckpoint(model_filepath, verbose=1,
                                                 save_weights_only=True, mode='min')
    csv_logger = keras.callbacks.CSVLogger(logs_filepath)

    nb_epoch = 200
    print(model.trainable_weights)
    model.load_weights('./mask_rcnn_coco.h5', by_name=True)

    # model.fit([train_X, train_x], train_y, epochs=200, batch_size=config.BATCH_SIZE, shuffle=True,
    #           callbacks=[checkpoint, csv_logger], validation_split=0.2)
    model.load_weights('models/model-02-3.54.h5')
    result = model.predict([train_X, train_x], batch_size=config.BATCH_SIZE, verbose=1)
    for sent in result:
    # for sent in train_y:
        print(decode_caption(sent, id_to_word))
