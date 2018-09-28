import os
import json
import keras
import pickle
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.models as KM
import skimage.io as skiimage_io
import skimage.color as skimage_color
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from utils import Dataset
from config import Config
from modified_dense_model import BatchNorm
from generate_one_roi_features import generate_features, load_model
from preprocess import encode_caption_v2, encode_word_v2, load_corpus, load_embeddings, tokenize_corpus, decode_word
from visualize import draw_boxes_and_captions
import skimage

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

    BATCH_SIZE = 64

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

    def add_sequences(self, sequences):
        self.sequences = sequences

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
        # captions = pad_sequences(caps, maxlen=self.padding_size, padding='post', dtype='float').astype(np.float32)
        rois = np.array(rois)
        captions = np.array(caps)
        return rois, captions

    def load_original_captions_and_rois(self, image_id):
        """ Returns string instance captions of the given image ID. """
        image_info = self.image_info[image_id]
        rois = np.array(image_info['rois'])
        captions = image_info['captions']
        return rois, captions

    def encode_region_caption(self, caption):
        """ Convert caption to word embedding vector """
        return encode_caption_v2(caption, self.word_to_id)


def load_sequences(dataset):
    sequences = []
    for _, image_id in zip(tqdm(list(range(len(dataset._image_ids)))), dataset._image_ids):
        _, captions = dataset.load_captions_and_rois(image_id)
        steps = captions.shape[0]
        for i in range(steps):
            sequences.append((image_id, i, [0],  np.argmax(captions[i][0])))
            for j in range(1, len(captions[i])):
                sequences.append((image_id, i, [np.argmax(cap) for cap in captions[i][:j]], np.argmax(captions[i][j])))
    return sequences


def build_model(features_shape, word_shape, config, units, inject=True):
    input_f = KL.Input(shape=features_shape, name='imgcap_features')
    features = KL.Conv2D(1024, (config.POOL_SIZE, config.POOL_SIZE), padding='valid', trainable=False,
                         name='mrcnn_class_conv1')(input_f)
    features = BatchNorm(axis=3, trainable=False, name='mrcnn_class_bn1')(features)
    features = KL.Activation('relu')(features)
    features = KL.Conv2D(1024, (1, 1), trainable=False, name='mrcnn_class_conv2')(features)
    features = BatchNorm(axis=3, trainable=False, name='mrcnn_class_bn2')(features)
    features = KL.Activation('relu')(features)
    features = KL.Lambda(lambda x: tf.squeeze(x, axis=[1]))(features)
    features = KL.Lambda(lambda x: tf.squeeze(x, axis=[1]))(features)
    if inject:
        features = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(features)

    input_w = KL.Input(shape=word_shape)
    word = KL.Embedding(input_dim=config.VOCABULARY_SIZE, output_dim=config.EMBEDDING_SIZE, trainable=False,
                        weights=[config.EMBEDDING_WEIGHTS], name='imgcap_embedding_layer', mask_zero=True)(input_w)
    word = KL.LSTM(1024)(word)
    if inject:
        word = KL.Lambda(lambda x: tf.expand_dims(x, axis=1))(word)

    result = KL.Concatenate()([features, word])
    if inject:
        result = KL.LSTM(units, return_sequences=False, name='imgcap_lstm')(result)
    result = KL.Dense(config.VOCABULARY_SIZE, activation='softmax', name='imgcap_d1')(result)

    return KM.Model(inputs=[input_f, input_w], outputs=result)


def data_generator(dataset, features_model, config, batch_size, shuffle=False):
    b = 0
    sequence_index = -1
    sequence_ids = np.array([i for i in range(len(dataset.sequences))])
    prev_im_id = -1
    prev_img_features = None
    while True:
        try:
            sequence_index = (sequence_index + 1) % len(sequence_ids)
            if shuffle and sequence_index == 0:
                np.random.shuffle(sequence_ids)
            sequence_id = sequence_ids[sequence_index]
            image_id, roi_id = dataset.sequences[sequence_id][0], dataset.sequences[sequence_id][1]
            prev_words, next_word = dataset.sequences[sequence_id][2], dataset.sequences[sequence_id][3]
            prev_word_features = pad_sequences([prev_words], config.PADDING_SIZE)[0]
            if prev_im_id == image_id:
                roi_features = prev_img_features[roi_id]
            else:
                img_features = generate_features(dataset, image_id, features_model)
                roi_features = img_features[roi_id]
                prev_img_features = img_features
            prev_im_id = image_id
            next_word_feature = np.zeros(config.VOCABULARY_SIZE)
            next_word_feature[next_word] = 1
            if b == 0:
                batch_image_features = np.zeros((batch_size,) + roi_features.shape, dtype=roi_features.dtype)
                batch_prev_words = np.zeros((batch_size,) +  prev_word_features.shape, dtype=prev_word_features.dtype)
                batch_next_word = np.zeros((batch_size,) + next_word_feature.shape, dtype=next_word_feature.dtype)
            batch_image_features[b] = roi_features
            batch_prev_words[b] = prev_word_features
            batch_next_word[b] = next_word_feature
            b += 1
            if b >= batch_size:
                yield [batch_image_features, batch_prev_words], batch_next_word
                b = 0
        except:
            raise Exception('An error occurred while processing sequence ' + str(sequence_id))


if __name__ == '__main__':
    train = True
    inject = True
    embeddings_file_path = '../dataset/glove.6B.300d.txt'

    data_directory = '../dataset/visual genome/'
    image_meta_file_path = '../dataset/image_data.json'
    data_file_path = '../dataset/region_descriptions.json'

    with open(image_meta_file_path, 'r', encoding='utf-8') as file:
        image_meta_data = json.loads(file.read())
    image_ids_list = [meta['image_id'] for meta in image_meta_data]

    train_val_image_ids = image_ids_list[:100000]
    test_image_ids = image_ids_list[100000:]

    id_to_word_file = '../dataset/id_to_word_v2.pickle'
    word_to_id_file = '../dataset/word_to_id_v2.pickle'
    embedding_matrix_file = '../dataset/embedding_matrix_v2.pickle'
    train_sequences_file = '../dataset/train_sequences.pickle'
    val_sequences_file = '../dataset/val_sequences.pickle'

    if not os.path.exists(id_to_word_file) or not os.path.exists(word_to_id_file) \
            or not os.path.exists(embedding_matrix_file):
        embeddings = load_embeddings(embeddings_file_path)
        tokens = tokenize_corpus(data_file_path, train_val_image_ids, embeddings)
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

    features_model = load_model()

    if inject:
        model_filepath = 'models/model1-90-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/text_generation_m1.log'
    else:
        model_filepath = 'models/model2-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/text_generation_m2.log'

    model = build_model((7, 7, 256), (10,), config, 256, inject)
    model.load_weights('mask_rcnn_coco.h5', by_name=True)
    print(model.summary())

    opt = keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy)

    if train:
        # Training dataset
        dataset_train = VisualGenomeDataset(word_to_id, config.PADDING_SIZE)
        dataset_train.load_visual_genome(data_directory, train_val_image_ids[75000:90000], image_meta_file_path,
                                         data_file_path)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = VisualGenomeDataset(word_to_id, config.PADDING_SIZE)
        dataset_val.load_visual_genome(data_directory, train_val_image_ids[90000:], image_meta_file_path,
                                       data_file_path)
        dataset_val.prepare()

        if not os.path.exists(train_sequences_file):
            train_sequences = load_sequences(dataset_train)
            with open(train_sequences_file, 'wb') as f:
                pickle.dump(train_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(train_sequences_file, 'rb') as f:
                train_sequences = pickle.load(f)
        if not os.path.exists(val_sequences_file):
            val_sequences = load_sequences(dataset_val)
            with open(val_sequences_file, 'wb') as f:
                pickle.dump(val_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(val_sequences_file, 'rb') as f:
                val_sequences = pickle.load(f)

        dataset_train.add_sequences(train_sequences)
        dataset_val.add_sequences(val_sequences)

        train_data_generator = data_generator(dataset_train, features_model, config, 1024)
        val_data_generator = data_generator(dataset_val, features_model, config, 1024)

        checkpoint = keras.callbacks.ModelCheckpoint(model_filepath, verbose=1, save_weights_only=True, mode='min')
        csv_logger = keras.callbacks.CSVLogger(logs_filepath)
        print(model.trainable_weights)
        print('Train on ' + str(len(train_sequences)) + ' samples')
        print('Validate on ' + str(len(val_sequences)) + ' samples')
        if inject:
            model.load_weights('models/model1-75-02-3.85.h5', by_name=True, skip_mismatch=True)
        else:
            model.load_weights('models/model2-85-4.01.h5', by_name=True, skip_mismatch=True)
        model.fit_generator(train_data_generator, epochs=2, steps_per_epoch=500,
                            callbacks=[checkpoint, csv_logger], validation_data=next(val_data_generator), verbose=1)
        #model.fit([train_X_f, train_X_w], train_y, epochs=200, batch_size=32, shuffle=True,
        #          callbacks=[checkpoint, csv_logger], validation_split=0.2)
    else:
        # Test dataset
        dataset_test = VisualGenomeDataset(word_to_id, config.PADDING_SIZE)
        dataset_test.load_visual_genome(data_directory, test_image_ids, image_meta_file_path,
                                         data_file_path)
        dataset_test.prepare()
        if inject:
            model.load_weights('models/model1-20-2.04.h5')
            model_name = 'm1'
        else:
            model.load_weights('models/model2-05-3.56.h5')
            model_name = 'm2'
        for im_id in dataset_test._image_ids[:15]:
            image_id = dataset_test.image_info[im_id]['id']
            print('\nImage: ' + str(image_id))
            features = generate_features(dataset_test, im_id, features_model)
            rois, captions = dataset_test.load_captions_and_rois(im_id)
            image = skimage.io.imread(os.path.join('../dataset/visual genome', str(image_id) + '.jpg'))
            for j in range(15):
                f = features[j]
                r = rois[j]
                c = captions[j]
                prev = [np.zeros(config.VOCABULARY_SIZE, dtype=np.float64)]
                for i in range(config.PADDING_SIZE - 1):
                    res = model.predict([np.array([f]), np.array([pad_sequences([[np.argmax(cap) for cap in prev]],
                                                                                config.PADDING_SIZE)[0]])])
                    prev.append(res[0])
                predicted_caption = ' '.join([decode_word(p, id_to_word) for p in prev])
                real_caption = ' '.join([decode_word(p, id_to_word) for p in c])
                print(predicted_caption, real_caption, sep='\t-\t')
                draw_boxes_and_captions(image, np.array([r]), np.array([predicted_caption]), '', image_id, j, model_name)
