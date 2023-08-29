import numpy as np
import pandas as pd
import gensim.downloader
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from torchtext.data import get_tokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tensorflow import keras
from keras import layers
import keras_tuner as kt
from keras.metrics import F1Score, Precision, Recall, Accuracy
from tqdm.auto import tqdm
import random
import re

from src.neo4j_database.graph_reddit import GraphReddit
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

graph = GraphReddit()
glove_wv = gensim.downloader.load('glove-wiki-gigaword-100')
# glove_wv = gensim.downloader.load('glove-twitter-100')


# glove-wiki-gigaword-50
# glove-wiki-gigaword-100
# glove-wiki-gigaword-200
# glove-wiki-gigaword-300
# glove-twitter-200
# glove-twitter-100
# glove-twitter-50
# glove-twitter-25


def embed_by_GloVe(doc: str, max_length: int = -1):
    '''
    Tokenise and embed the sentence with GloVe

    Args:
        doc: The sentence needed to be embedded.
        max_length: The max length of the return word embedding list
    Returns:
        A list of word embedding.
    '''
    tokenizers = get_tokenizer('basic_english')
    tokenized_doc = tokenizers(doc)
    embedding = []
    for word in tokenized_doc:
        if word in glove_wv:
            embedding.append(glove_wv[word])
        if max_length != -1:
            if len(embedding) == max_length:
                break

    return np.array(embedding)


def pad_embedding(embedding: np.array, max_length: int):
    '''
    Pad or truncate the word embedding list to 'max_length'.

    Args:
        embedding: The word embedding list
        max_length: The max length which would be used to pad or truncate the word embedding list

    Returns:
        embedding: In the form of 'numpy.array'
    '''
    length = embedding.shape[0]
    if length > max_length:
        embedding = embedding[:max_length]
    elif length < max_length:
        embedding = np.concatenate((embedding, np.zeros(((max_length - length), embedding.shape[1]), dtype=float)))

    return np.array(embedding, dtype=float)


def clean_text(text: str):
    mention_regex = r'@[a-zA-Z0-9]*'
    url_regex = r'[\[]*[a-zA-Z@ ]*[\](]*http[a-zA-Z0-9.\/?&:%=\-\[\]\)]*'
    text = re.sub(mention_regex, ' ', text)
    text = re.sub(url_regex, ' ', text)
    return text


def get_all_information(reddit_ids: np.array):
    get_all_information_query = '''
        MATCH (r:Reddit)
        WHERE id(r)=$reddit_id
        WITH r
        OPTIONAL MATCH (r)-[:HAS_COMMENT]->(c:Comment)
        WHERE c.text <> '[removed]' and c.text <> '[deleted]'
        OPTIONAL MATCH (r)-[:HAS_IMAGE]->(i1:Image)
        WHERE i1.width is not null
        OPTIONAL MATCH (r)-[:HAS_ARTICLE]->(a:Article)
        OPTIONAL MATCH (a)-[:HAS_TOP_IMAGE]->(i2:Image)
        WHERE i2.width is not null
        RETURN r.ups as reddit_ups, r.title as reddit_title, r.text as reddit_text, 
            r.commentCount as reddit_comment_count, a.content as article_content, c.text as comment_text, 
            c.ups as comment_ups, c.emotion as comment_emotion, i1 as reddit_image, i2 as article_image
    '''

    information_lists = []

    for reddit_id in tqdm(reddit_ids, desc='Retrieving reddit information'):
        reddit_features_dict = graph.query(get_all_information_query, reddit_id=reddit_id).to_dict('records')
        # Reddit content: title+text
        reddit_content = reddit_features_dict[0]['reddit_title']
        if reddit_features_dict[0]['reddit_text'] is not None:
            reddit_content += ' ' + reddit_features_dict[0]['reddit_text']
        # Clean the data
        reddit_content = clean_text(reddit_content)

        # Article content
        article_content = None
        if reddit_features_dict[0]['article_content'] is not None:
            article_content = reddit_features_dict[0]['article_content']

        # Reddit feature
        reddit_feature = np.array([reddit_features_dict[0]['reddit_ups']], dtype=float)

        # Image features
        image_features = None
        image = reddit_features_dict[0]['reddit_image'] if reddit_features_dict[0]['article_image'] is None else \
            reddit_features_dict[0]['article_image']
        if image is not None:
            image_features = np.array([image['width'], image['height'], image['size'],
                                       image['r'], image['g'], image['b']], dtype=float)

        # Comment content
        comment_content = None
        if len(reddit_features_dict) > 1 or reddit_features_dict[0]['comment_text'] is not None:
            comment_content = []
            for row in reddit_features_dict:
                text = clean_text(row['comment_text'])
                if text != ' ':
                    comment_content.append({"text": text, "emotion": row['comment_emotion']})

        all_information = {'reddit_id': reddit_id, 'reddit_content': reddit_content,
                           'article_content': article_content, 'comment_content': comment_content,
                           'reddit_feature': reddit_feature, 'image_features': image_features}
        information_lists.append(all_information)

    return information_lists


def embed_data(information_lists: list, max_content_length: int = 500, max_context_length: int = 500,
               with_context_info: bool = True):
    # Emotion labels (See: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
    emotion_dict = dict(anger=1, disgust=2, fear=3, joy=4, neutral=5, sadness=6, surprise=7)
    data = []
    content_text_emb_dim_count = []
    for info in tqdm(information_lists, desc='Embedding textual data by GloVe'):
        # Embedding textual content
        text_word_emb = embed_by_GloVe(info['reddit_content'])
        if text_word_emb.size == 0:
            text_word_emb = embed_by_GloVe('no title')
        if info['article_content'] is not None:
            article_emb = embed_by_GloVe(info['article_content'])
            text_word_emb = np.concatenate((text_word_emb, article_emb))
        comment_emb = None
        comment_feature = []
        if with_context_info and info['comment_content'] is not None:
            for c in info['comment_content']:
                comment_word_emb = embed_by_GloVe(c['text'], 100)
                if comment_word_emb.size == 0:
                    comment_word_emb = np.zeros((100, text_word_emb.shape[1]), dtype=float)
                comment_word_emb = pad_embedding(comment_word_emb, 100)
                comment_emb = comment_word_emb if comment_emb is None \
                    else np.concatenate((comment_emb, comment_word_emb))
                comment_feature.append(emotion_dict[c['emotion']])
                if len(comment_feature) == 5:
                    break
                # comment_text = c['emotion'] + ' ' + c['text']
                # text_word_emb = np.concatenate((text_word_emb, embed_by_GloVe(comment_text, 150)))
            if len(comment_feature) < 5:
                comment_feature += [0] * (5 - len(comment_feature))

        if comment_emb is None:
            comment_emb = np.zeros((max_context_length, text_word_emb.shape[1]), dtype=float)
        if len(comment_feature) == 0:
            comment_feature = np.zeros(5, dtype=float)
        comment_feature = np.array(comment_feature, dtype=float)

        # Padding and truncation
        content_text_emb_dim_count.append(text_word_emb.shape[0])
        text_word_emb = pad_embedding(text_word_emb, max_length=max_content_length)
        comment_emb = pad_embedding(comment_emb, max_context_length)

        # Explicit features
        context_features = info['reddit_feature'] if with_context_info else np.array([0], dtype=float)
        if info['image_features'] is not None:
            content_features = info['image_features']
        else:
            content_features = np.zeros(6, dtype=float)

        context_features = np.append(context_features, comment_feature)
        content_features = content_features.reshape((-1, 1))
        context_features = context_features.reshape((-1, 1))

        data.append({'text_emb': text_word_emb, 'comment_emb': comment_emb,
                     # 'features': features,
                     'content_features': content_features,
                     'context_features': context_features, 'reddit_id': info['reddit_id']})
    content_text_avg_rows = np.average(np.array(content_text_emb_dim_count))
    # print(f'\nContent text average row count: {content_text_avg_rows}')
    # print(f'Content text maximum row count: {max(content_text_emb_dim_count)}')
    # print(f'Content text minimum row count: {min(content_text_emb_dim_count)}')

    return data, content_text_avg_rows


def train_val_test_split(dataset_size: str = 'large'):
    get_mis_reddit_query = '''
        MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]->(c:Claim)-[:HAS_LABEL]->(l:Label)
        WHERE l.verdict='misinformation' AND l.size=$dataset_size
        RETURN id(r) as reddit_ids
    '''

    get_fac_reddit_query = '''
        MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]->(c:Claim)-[:HAS_LABEL]->(l:Label)
        WHERE l.verdict='factual' AND l.size=$dataset_size
        RETURN id(r) as reddit_ids
    '''

    # Split the dataset into train (70%) and test (30%) sets
    # such that all contain the same distribution of classes, or as close as possible.
    fac_reddit_df = graph.query(get_fac_reddit_query, dataset_size=dataset_size)
    mis_reddit_df = graph.query(get_mis_reddit_query, dataset_size=dataset_size)

    fac_reddit_ids = fac_reddit_df.reddit_ids.to_numpy()
    mis_reddit_ids = mis_reddit_df.reddit_ids.to_numpy()

    # Calculate the class weight for applying class weights in the model training phase
    mis = np.ones(mis_reddit_ids.shape, dtype=float)
    fac = np.zeros(fac_reddit_ids.shape, dtype=float)
    train_all_classes = np.append(mis, fac)
    weights = compute_class_weight('balanced', classes=np.unique(train_all_classes), y=train_all_classes)
    class_weights = {0: weights[0], 1: weights[1]}

    mis_reddit_train_x, mis_reddit_test_x = train_test_split(mis_reddit_ids, test_size=0.3)
    fac_reddit_train_x, fac_reddit_test_x = train_test_split(fac_reddit_ids, test_size=0.3)

    reddit_test = {'x': np.append(mis_reddit_test_x, fac_reddit_test_x),
                   'y': np.append(np.ones(mis_reddit_test_x.shape, dtype=float),
                                  np.zeros(fac_reddit_test_x.shape, dtype=float))}

    reddit_train = {'x': np.append(mis_reddit_train_x, fac_reddit_train_x),
                    'y': np.append(np.ones(mis_reddit_train_x.shape, dtype=float),
                                   np.zeros(fac_reddit_train_x.shape, dtype=float))}

    reddit_val = {}
    # Split the train set into train (80%) and validation (20%) sets
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(reddit_train['x'], reddit_train['y'])):
        train_data_x = reddit_train['x'][train_idx]
        train_data_y = reddit_train['y'][train_idx]
        val_data_x = reddit_train['x'][val_idx]
        val_data_y = reddit_train['y'][val_idx]
        reddit_train['x'] = train_data_x
        reddit_train['y'] = train_data_y
        reddit_val['x'] = val_data_x
        reddit_val['y'] = val_data_y
        break
    reddit_val_list = [{'x': x, 'y': y} for (x, y) in zip(reddit_val['x'], reddit_val['y'])]
    random.shuffle(reddit_val_list)
    reddit_val_x = np.array([r['x'] for r in reddit_val_list])
    reddit_val_y = np.array([r['y'] for r in reddit_val_list])
    reddit_val = {'x': reddit_val_x, 'y': reddit_val_y}

    # Shuffle the dataset
    reddit_train_list = [{'x': x, 'y': y} for (x, y) in zip(reddit_train['x'], reddit_train['y'])]
    random.shuffle(reddit_train_list)
    reddit_train_x = np.array([r['x'] for r in reddit_train_list])
    reddit_train_y = np.array([r['y'] for r in reddit_train_list])
    reddit_train = {'x': reddit_train_x, 'y': reddit_train_y}
    reddit_test_list = [{'x': x, 'y': y} for (x, y) in zip(reddit_test['x'], reddit_test['y'])]
    random.shuffle(reddit_test_list)
    reddit_test_x = np.array([r['x'] for r in reddit_test_list])
    reddit_test_y = np.array([r['y'] for r in reddit_test_list])
    reddit_test = {'x': reddit_test_x, 'y': reddit_test_y}

    print(f'Train: {reddit_train["x"].shape[0]}; Val: {reddit_val["x"].shape[0]}; Test: {reddit_test["x"].shape[0]}.')

    return reddit_train, reddit_val, reddit_test, class_weights


# Concatenate the content and context embeddings of each reddit in the reddit_ids list
def concatenate_embedding_from_db(reddit_ids: np.array):
    concatenated_embedding = []
    emotion_dict = dict(anger=1, disgust=2, fear=3, joy=4, neutral=5, sadness=6, surprise=7)

    for idx, reddit_id in enumerate(reddit_ids):
        get_all_embedding_query = '''
            MATCH (r:Reddit)
            WHERE id(r)=$reddit_id
            OPTIONAL MATCH (r)-[:HAS_COMMENT]->(c:Comment)
            OPTIONAL MATCH (r)-[:HAS_IMAGE]->(i1:Image)
            WHERE i1.width is not null
            OPTIONAL MATCH (r)-[:HAS_ARTICLE]->(a:Article)
            OPTIONAL MATCH (a)-[:HAS_TOP_IMAGE]->(i2:Image)
            WHERE i2.width is not null
            RETURN r.ups as reddit_ups, r.embedding as reddit_embedding, a.embedding as article_embedding, 
                c.embedding as comment_embedding, c.emotion as comment_emotion, 
                i1 as reddit_image, i2 as article_image
        '''
        # add image features
        embedding_dict = graph.query(get_all_embedding_query, reddit_id=reddit_id).to_dict('records')
        content_embedding = embedding_dict[0]['reddit_embedding'] if embedding_dict[0]['article_embedding'] is None \
            else embedding_dict[0]['article_embedding']

        # Image features
        image = embedding_dict[0]['reddit_image'] if embedding_dict[0]['article_image'] is None else \
            embedding_dict[0]['article_image']
        if image is not None:
            features = [image['width'], image['height'], image['size'], image['r'], image['g'], image['b']]
            content_embedding += features

        context_embedding = []
        if len(embedding_dict) > 1 or embedding_dict[0]['comment_embedding'] is not None:
            for row in embedding_dict:
                emb = row['comment_embedding']
                emotion = [float(emotion_dict[row['comment_emotion']])]
                emb += emotion
                context_embedding += emb

        # ups count
        context_embedding += [float(embedding_dict[0]['reddit_ups'])]

        all_embedding = content_embedding + context_embedding
        # Padding on the right side
        all_embedding = all_embedding + [0.0] * (1050 - len(all_embedding)) if (len(all_embedding) < 1050) \
            else all_embedding[:1050]
        # all_embedding = np.array(all_embedding, dtype=float).tolist()
        concatenated_embedding.append(all_embedding)
        print(idx)

    return concatenated_embedding


def data_preprocessing_for_ml_classifier(x):
    content_x = x[0]
    comment_x = x[-1]
    # Flatten the word embeddings
    content_x = content_x.reshape(content_x.shape[0], -1)
    comment_x = comment_x.reshape(comment_x.shape[0], -1)
    # Dimensionality reduction
    pca_model = PCA(n_components=500)
    pca_model.fit(content_x)
    print("Sum of variance ratios: ", sum(pca_model.explained_variance_ratio_))
    content_x_comps = pca_model.transform(content_x)
    pca_model.fit(comment_x)
    print("Sum of variance ratios: ", sum(pca_model.explained_variance_ratio_))
    comment_x_comps = pca_model.transform(comment_x)
    x_comps = np.array([np.append(content_x_comps[i], comment_x_comps[i])
                        for i in range(content_x_comps.shape[0])])
    return x_comps


def train_ML_text_model(train_x, test_x, train_y, test_y):
    train_x_comps = data_preprocessing_for_ml_classifier(train_x)
    test_x_comps = data_preprocessing_for_ml_classifier(test_x)

    # Logistic regression
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(train_x_comps, train_y)
    lr_pred_y = lr.predict(test_x_comps)
    acc = lr.score(test_x_comps, test_y)
    print(f'Performance of Logistic Regression [f1, precision, recall, accuracy]: '
          f'{[f1_score(test_y, lr_pred_y), precision_score(test_y, lr_pred_y), recall_score(test_y, lr_pred_y), acc]}')
    tn, fp, fn, tp = confusion_matrix(test_y, lr_pred_y).ravel()
    print(f'Confusion matrix (tn, fp, fn, tp): {(tn, fp, fn, tp)}')

    # DecisionTreeClassifier
    dt = DecisionTreeClassifier(class_weight='balanced')
    dt.fit(train_x_comps, train_y)
    dt_pred_y = dt.predict(test_x_comps)
    acc = dt.score(test_x_comps, test_y)
    print(f'Performance of DecisionTreeClassifier [f1, precision, recall, accuracy]: '
          f'{[f1_score(test_y, dt_pred_y), precision_score(test_y, dt_pred_y), recall_score(test_y, dt_pred_y), acc]}')
    tn, fp, fn, tp = confusion_matrix(test_y, dt_pred_y).ravel()
    print(f'Confusion matrix (tn, fp, fn, tp): {(tn, fp, fn, tp)}')

    # Random forest
    rf = RandomForestClassifier(class_weight='balanced')
    rf.fit(train_x_comps, train_y)
    rf_pred_y = rf.predict(test_x_comps)
    acc = rf.score(test_x_comps, test_y)
    print(f'Performance of Random Forest [f1, precision, recall, accuracy]: '
          f'{[f1_score(test_y, rf_pred_y), precision_score(test_y, rf_pred_y), recall_score(test_y, rf_pred_y), acc]}')
    tn, fp, fn, tp = confusion_matrix(test_y, rf_pred_y).ravel()
    print(f'Confusion matrix (tn, fp, fn, tp): {(tn, fp, fn, tp)}')

    # SVM
    svm = SVC(class_weight='balanced')
    svm.fit(train_x_comps, train_y)
    svm_pred_y = svm.predict(test_x_comps)
    acc = svm.score(test_x_comps, test_y)
    print(f'Performance of SVM [f1, precision, recall, accuracy]: '
          f'{[f1_score(test_y, svm_pred_y), precision_score(test_y, svm_pred_y), recall_score(test_y, svm_pred_y), acc]}')
    tn, fp, fn, tp = confusion_matrix(test_y, svm_pred_y).ravel()
    print(f'Confusion matrix (tn, fp, fn, tp): {(tn, fp, fn, tp)}')




def create_CNN_baseline_model(dropout: float = 0.2, learning_rate: float = 0.001):
    # Processing text input
    # Model for processing textual information
    text_input = keras.layers.Input(shape=(500, 100))
    comment_input = keras.layers.Input(shape=(500, 100))
    concat_text = keras.layers.Concatenate(axis=1)([text_input, comment_input])
    text_conv1d_1 = keras.layers.Conv1D(128, 5, activation='relu')(concat_text)  # 996, 128
    text_maxpool1d_1 = keras.layers.MaxPool1D(5, padding='same')(text_conv1d_1)  # 200, 128

    # Model for processing other features
    content_feature_input = keras.layers.Input(shape=(6, 1))
    context_feature_input = keras.layers.Input(shape=(6, 1))
    feature_concat = keras.layers.Concatenate(axis=1)([content_feature_input, context_feature_input])
    feature_dense = keras.layers.Dense(128)(feature_concat)

    # Concatenate text and feature embeddings
    concat_input = keras.layers.Concatenate(axis=1)([text_maxpool1d_1, feature_dense])
    conv1d_1 = keras.layers.Conv1D(128, 5)(concat_input)  # 208, 128
    maxpool1d_1 = keras.layers.MaxPool1D(35, padding='same')(conv1d_1)  # 6, 128
    flatten = keras.layers.Flatten()(maxpool1d_1)  # 768
    dropout_1 = keras.layers.Dropout(dropout)(flatten)
    dense_1 = keras.layers.Dense(128)(dropout_1)  # 128
    bn = keras.layers.BatchNormalization()(dense_1)
    activation = keras.layers.Activation('relu')(bn)
    dropout_2 = keras.layers.Dropout(dropout)(activation)
    output = keras.layers.Dense(1, activation='sigmoid')(dropout_2)

    model = keras.Model(inputs=[text_input, content_feature_input,
                                context_feature_input, comment_input], outputs=[output])
    model.summary()

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
    ]

    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=METRICS)

    return model


def create_LSTM_baseline_model(dropout: float = 0.2, learning_rate: float = 0.001):
    # Processing text input
    # Model for processing textual information
    text_input = keras.layers.Input(shape=(500, 100))
    comment_input = keras.layers.Input(shape=(500, 100))
    concat_text = keras.layers.Concatenate(axis=1)([text_input, comment_input])
    text_conv1d_1 = keras.layers.Conv1D(128, 5, activation='relu')(concat_text)  # 996, 128
    text_maxpool1d_1 = keras.layers.MaxPool1D(5, padding='same')(text_conv1d_1)  # 200, 128

    # Model for processing other features
    content_feature_input = keras.layers.Input(shape=(6, 1))
    context_feature_input = keras.layers.Input(shape=(6, 1))
    feature_concat = keras.layers.Concatenate(axis=1)([content_feature_input, context_feature_input])
    feature_dense = keras.layers.Dense(128)(feature_concat)

    # Concatenate text and feature embeddings
    concat_input = keras.layers.Concatenate(axis=1)([text_maxpool1d_1, feature_dense])
    lstm_1 = keras.layers.LSTM(128)(concat_input)  # 256
    # dense_1 = keras.layers.Dense(128)(lstm_1)  # 128
    bn = keras.layers.BatchNormalization()(lstm_1)
    activation = keras.layers.Activation('relu')(bn)
    dropout_2 = keras.layers.Dropout(dropout)(activation)
    output = keras.layers.Dense(1, activation='sigmoid')(dropout_2)

    model = keras.Model(inputs=[text_input, content_feature_input,
                                context_feature_input, comment_input], outputs=[output])
    model.summary()

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
    ]

    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=METRICS)

    return model


def create_model_for_classification_split_comment(hp, dropout: float = 0.2, learning_rate: float = 0.001):
    if hp is not None:
        dropout = hp.Float('dropout', 0.1, 0.5, step=0.1, default=0.2)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Model for processing textual information
    text_input = keras.layers.Input(shape=(500, 100))
    # text_input = keras.layers.Input(shape=(500, 50))
    # text_input = keras.layers.Input(shape=(500, 25))
    text_conv1d_1 = keras.layers.Conv1D(128, 3)(text_input)  # 498, 128
    text_maxpool1d_1 = keras.layers.MaxPool1D(5, padding='same')(text_conv1d_1)  # 100, 128
    text_conv1d_2 = keras.layers.Conv1D(128, 4)(text_input)  # 497, 128
    text_maxpool1d_2 = keras.layers.MaxPool1D(5, padding='same')(text_conv1d_2)  # 100, 128
    text_conv1d_3 = keras.layers.Conv1D(128, 5)(text_input)  # 496, 128
    text_maxpool1d_3 = keras.layers.MaxPool1D(5, padding='same')(text_conv1d_3)  # 100, 128

    # Model for processing content features
    content_feature_input = keras.layers.Input(shape=(6, 1))
    content_feature_dense = keras.layers.Dense(128)(content_feature_input)  # 6, 128
    content_feature_bn = keras.layers.BatchNormalization()(content_feature_dense)
    content_feature = keras.layers.Activation('relu')(content_feature_bn)

    # Concatenate three different text embeddings
    concat_content = keras.layers.Concatenate(axis=1)([text_maxpool1d_1, text_maxpool1d_2,
                                                       text_maxpool1d_3, content_feature])  # 306, 128

    # Model for processing comment embeddings
    comment_input = keras.layers.Input(shape=(500, 100))
    # comment_input = keras.layers.Input(shape=(500, 50))
    # comment_input = keras.layers.Input(shape=(500, 25))
    comment_conv1d_1 = keras.layers.Conv1D(128, 3)(comment_input)  # 498, 128
    comment_maxpool1d_1 = keras.layers.MaxPool1D(5, padding='same')(comment_conv1d_1)  # 100, 128
    comment_conv1d_2 = keras.layers.Conv1D(128, 4)(comment_input)  # 498, 128
    comment_maxpool1d_2 = keras.layers.MaxPool1D(5, padding='same')(comment_conv1d_2)  # 100, 128
    comment_conv1d_3 = keras.layers.Conv1D(128, 5)(comment_input)  # 498, 128
    comment_maxpool1d_3 = keras.layers.MaxPool1D(5, padding='same')(comment_conv1d_3)  # 100, 128

    # Model for processing other features
    context_feature_input = keras.layers.Input(shape=(6, 1))
    context_feature_dense = keras.layers.Dense(128)(context_feature_input)  # 6, 128
    context_feature_bn = keras.layers.BatchNormalization()(context_feature_dense)
    context_feature = keras.layers.Activation('relu')(context_feature_bn)

    # Concatenate three different comment embeddings
    concat_context = keras.layers.Concatenate(axis=1)([comment_maxpool1d_1, comment_maxpool1d_2,
                                                       comment_maxpool1d_3, context_feature])  # 306, 128

    # Concatenate text and feature embeddings
    concat = keras.layers.Concatenate(axis=1)([concat_content, concat_context])  # 612, 128
    conv1d_1 = keras.layers.Conv1D(128, 5)(concat)  # 608, 128
    maxpool1d_1 = keras.layers.MaxPool1D(5, padding='same')(conv1d_1)  # 122, 128
    conv1d_2 = keras.layers.Conv1D(128, 5)(maxpool1d_1)  # 118, 128
    maxpool1d_2 = keras.layers.MaxPool1D(30, padding='same')(conv1d_2)  # 4, 128
    flatten = keras.layers.Flatten()(maxpool1d_2)  # 512
    dropout_1 = keras.layers.Dropout(dropout)(flatten)
    dense_1 = keras.layers.Dense(128)(dropout_1)  # 128
    bn_2 = keras.layers.BatchNormalization()(dense_1)
    activation_2 = keras.layers.Activation('relu')(bn_2)
    dropout_2 = keras.layers.Dropout(dropout)(activation_2)
    output = keras.layers.Dense(1, activation='sigmoid')(dropout_2)

    model = keras.Model(inputs=[text_input, content_feature_input,
                                context_feature_input, comment_input], outputs=[output])
    # model.summary()

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
    ]

    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=METRICS)

    return model


def create_model_for_classification_without_content(dropout: float = 0.2, learning_rate: float = 0.001):
    # Model for processing textual information
    text_input = keras.layers.Input(shape=(500, 100))
    # Model for processing content features
    content_feature_input = keras.layers.Input(shape=(6, 1))

    # Model for processing comment embeddings
    comment_input = keras.layers.Input(shape=(500, 100))
    comment_conv1d_1 = keras.layers.Conv1D(128, 3)(comment_input)  # 498, 128
    comment_maxpool1d_1 = keras.layers.MaxPool1D(5, padding='same')(comment_conv1d_1)  # 100, 128
    comment_conv1d_2 = keras.layers.Conv1D(128, 4)(comment_input)  # 498, 128
    comment_maxpool1d_2 = keras.layers.MaxPool1D(5, padding='same')(comment_conv1d_2)  # 100, 128
    comment_conv1d_3 = keras.layers.Conv1D(128, 5)(comment_input)  # 498, 128
    comment_maxpool1d_3 = keras.layers.MaxPool1D(5, padding='same')(comment_conv1d_3)  # 100, 128

    # Model for processing other features
    context_feature_input = keras.layers.Input(shape=(6, 1))
    context_feature_dense = keras.layers.Dense(128)(context_feature_input)  # 6, 128
    context_feature_bn = keras.layers.BatchNormalization()(context_feature_dense)
    context_feature = keras.layers.Activation('relu')(context_feature_bn)

    # Concatenate text and feature embeddings
    concat_context = keras.layers.Concatenate(axis=1)([comment_maxpool1d_1, comment_maxpool1d_2,
                                                       comment_maxpool1d_3, context_feature])  # 306, 128
    conv1d_1 = keras.layers.Conv1D(128, 5)(concat_context)  # 302, 128
    maxpool1d_1 = keras.layers.MaxPool1D(5, padding='same')(conv1d_1)  # 61, 128
    conv1d_2 = keras.layers.Conv1D(128, 5)(maxpool1d_1)  # 57, 128
    maxpool1d_2 = keras.layers.MaxPool1D(20, padding='same')(conv1d_2)  # 3, 128
    flatten = keras.layers.Flatten()(maxpool1d_2)  # 384
    dropout_1 = keras.layers.Dropout(dropout)(flatten)
    dense_1 = keras.layers.Dense(128)(dropout_1)  # 128
    bn_2 = keras.layers.BatchNormalization()(dense_1)
    activation_2 = keras.layers.Activation('relu')(bn_2)
    dropout_2 = keras.layers.Dropout(dropout)(activation_2)
    output = keras.layers.Dense(1, activation='sigmoid')(dropout_2)

    model = keras.Model(inputs=[text_input, content_feature_input,
                                context_feature_input, comment_input], outputs=[output])
    # model.summary()

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
    ]

    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=METRICS)

    return model


def create_model_for_classification_without_context(dropout: float = 0.2, learning_rate: float = 0.001):
    # Model for processing textual information
    text_input = keras.layers.Input(shape=(500, 100))
    text_conv1d_1 = keras.layers.Conv1D(128, 3)(text_input)  # 498, 128
    text_maxpool1d_1 = keras.layers.MaxPool1D(5, padding='same')(text_conv1d_1)  # 100, 128
    text_conv1d_2 = keras.layers.Conv1D(128, 4)(text_input)  # 497, 128
    text_maxpool1d_2 = keras.layers.MaxPool1D(5, padding='same')(text_conv1d_2)  # 100, 128
    text_conv1d_3 = keras.layers.Conv1D(128, 5)(text_input)  # 496, 128
    text_maxpool1d_3 = keras.layers.MaxPool1D(5, padding='same')(text_conv1d_3)  # 100, 128

    # Model for processing content features
    content_feature_input = keras.layers.Input(shape=(6, 1))
    content_feature_dense = keras.layers.Dense(128)(content_feature_input)  # 6, 128
    content_feature_bn = keras.layers.BatchNormalization()(content_feature_dense)
    content_feature = keras.layers.Activation('relu')(content_feature_bn)

    # Concatenate three different text embeddings
    concat_content = keras.layers.Concatenate(axis=1)([text_maxpool1d_1, text_maxpool1d_2,
                                                       text_maxpool1d_3, content_feature])  # 306, 128
    conv1d_1 = keras.layers.Conv1D(128, 5)(concat_content)  # 302, 128
    maxpool1d_1 = keras.layers.MaxPool1D(5, padding='same')(conv1d_1)  # 61, 128
    conv1d_2 = keras.layers.Conv1D(128, 5)(maxpool1d_1)  # 57, 128
    maxpool1d_2 = keras.layers.MaxPool1D(20, padding='same')(conv1d_2)  # 3, 128
    flatten = keras.layers.Flatten()(maxpool1d_2)  # 384
    dropout_1 = keras.layers.Dropout(dropout)(flatten)
    dense_1 = keras.layers.Dense(128)(dropout_1)  # 128
    bn_2 = keras.layers.BatchNormalization()(dense_1)
    activation_2 = keras.layers.Activation('relu')(bn_2)
    dropout_2 = keras.layers.Dropout(dropout)(activation_2)
    output = keras.layers.Dense(1, activation='sigmoid')(dropout_2)

    context_feature_input = keras.layers.Input(shape=(6, 1))
    comment_input = keras.layers.Input(shape=(500, 100))

    model = keras.Model(inputs=[text_input, content_feature_input,
                                context_feature_input, comment_input], outputs=[output])
    model.summary()

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
    ]

    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=METRICS)

    return model


def fit_model(train_x, val_x, train_y, val_y, epochs: int = 30,
              batch_size: int = 128, class_weights: np.array = None,
              tuning_mode: bool = False, with_context: bool = True,
              dropout: float = 0.2, learning_rate: float = 0.001):
    # set early stopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # define the model checkpoint callback -> this will keep on saving the model as a physical file
    model_checkpoint = ModelCheckpoint('misinformation_detector_bs_32.h5', verbose=1, save_best_only=True)

    if tuning_mode:
        # Tuning mode
        tuner = kt.Hyperband(create_model_for_classification_split_comment,
                             objective='val_loss',
                             max_epochs=50,
                             factor=3,
                             directory='./',
                             project_name='Keras_tuning_' + str(batch_size))
        tuner.search(train_x, train_y, epochs=epochs, batch_size=batch_size,
                     class_weight=class_weights,
                     callbacks=[early_stopping, model_checkpoint],
                     validation_data=(val_x, val_y))
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"""
        Batch size={batch_size}
        The hyperparameter search is complete. The optimal dropout rate is {best_hps.get('dropout')}
        and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
        """)
        return best_hps, tuner

    # Training mode
    if with_context:
        model = create_model_for_classification_split_comment(hp=None, dropout=dropout, learning_rate=learning_rate)
        # model = create_CNN_baseline_model(dropout=dropout, learning_rate=learning_rate)
        # model = create_LSTM_baseline_model(dropout=dropout, learning_rate=learning_rate)
        # model = create_model_for_classification_without_content(dropout, learning_rate)
    else:
        model = create_model_for_classification_without_context(dropout=dropout, learning_rate=learning_rate)
    results = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        class_weight=class_weights,
                        callbacks=[early_stopping, model_checkpoint],
                        validation_data=(val_x, val_y))

    return results, model


def create_classifer_with_sentence_embedding(input_idm: int = 1050, dropout: float = 0.1):
    model = keras.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(None, input_idm)))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    metrics = [Accuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), F1Score(name='f1score')]

    model.compile(optimizer=keras.optimizers.Adam(3e-5),
                  loss=keras.losses.binary_crossentropy,
                  metrics=metrics)

    return model


def fit_model_se(train_x, val_x, train_y, val_y, epochs: int = 30, batch_size: int = 64):
    # set early stopping criteria
    pat = 5  # this is the number of epochs with no improvment after which the training will stop
    early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

    # define the model checkpoint callback -> this will keep on saving the model as a physical file
    model_checkpoint = ModelCheckpoint('misinformation_detector_1.h5', verbose=1, save_best_only=True)

    model = create_classifer_with_sentence_embedding()
    results = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint])
    print("Val score: ", model.evaluate(val_x, val_y)[1:])
    val_pre_y = model.predict(val_x)
    val_pre_y1 = model.predict(val_x)
    val_pre_y = np.array([1 if y >= 0.5 else 0 for y in val_pre_y], dtype=float)
    print("Val score: ", model.evaluate(val_x, val_pre_y)[1:])
    return results


def train_classifer_with_sentence_embedding():
    batch_size = 32
    epochs = 30
    train, val, test, _ = train_val_test_split(dataset_size='large')
    print(f'Train size: {len(train["x"])}')
    print(f'Val size: {len(val)}')
    train_x = np.array(concatenate_embedding_from_db(train['x']))
    train_y = train['y']
    val_x = np.array(concatenate_embedding_from_db(val['x']))
    val_y = val['y']
    fit_model_se(train_x, val_x, train_y, val_y, epochs, batch_size)
    print("=======" * 12, end="\n\n\n")


def train_model(dataset_size: str = 'large'):
    train, val, test, class_weights = train_val_test_split(dataset_size)

    # Data Processing
    train_data_info = get_all_information(train['x'])
    val_data_info = get_all_information(val['x'])
    test_data_info = get_all_information(test['x'])

    graph.close()
    train_emb, train_avg_content_text_rows = embed_data(train_data_info, max_content_length=500, max_context_length=500,
                                                        with_context_info=True)
    val_emb, val_avg_content_text_rows = embed_data(val_data_info, max_content_length=500, max_context_length=500,
                                                    with_context_info=True)
    test_emb, test_avg_content_text_rows = embed_data(test_data_info, max_content_length=500, max_context_length=500,
                                                      with_context_info=True)

    train_text_emb = np.array([emb['text_emb'] for emb in train_emb])
    train_comment_emb = np.array([emb['comment_emb'] for emb in train_emb])
    train_content_feature_emb = np.array([emb['content_features'] for emb in train_emb])
    train_context_feature_emb = np.array([emb['context_features'] for emb in train_emb])
    train_input = [train_text_emb, train_content_feature_emb, train_context_feature_emb, train_comment_emb]
    val_text_emb = np.array([emb['text_emb'] for emb in val_emb])
    val_comment_emb = np.array([emb['comment_emb'] for emb in val_emb])
    val_content_feature_emb = np.array([emb['content_features'] for emb in val_emb])
    val_context_feature_emb = np.array([emb['context_features'] for emb in val_emb])
    val_input = [val_text_emb, val_content_feature_emb, val_context_feature_emb, val_comment_emb]
    test_text_emb = np.array([emb['text_emb'] for emb in test_emb])
    test_comment_emb = np.array([emb['comment_emb'] for emb in test_emb])
    test_content_feature_emb = np.array([emb['content_features'] for emb in test_emb])
    test_context_feature_emb = np.array([emb['context_features'] for emb in test_emb])
    test_input = [test_text_emb, test_content_feature_emb, test_context_feature_emb, test_comment_emb]

    # Train the final model
    epochs = 50
    # batch_size_list = [8, 16, 32, 64, 128, 256]  # 8, 16, 32, 64, 128, 256, 512
    batch_size_list = [32]
    dropout = 0.1
    learning_rate = 0.001
    test_results = []
    val_results = []
    tuning_mode = False
    best_hp = []
    for batch_size in batch_size_list:
        result, model = fit_model(train_input, val_input, train['y'], val['y'],
                                  epochs=epochs, batch_size=batch_size, class_weights=class_weights,
                                  tuning_mode=tuning_mode, with_context=False, dropout=dropout,
                                  learning_rate=learning_rate)
        if tuning_mode:
            best_hp.append({'dropout': result.get('dropout'), 'lr': result.get('learning_rate')})
            hypermodel = model.hypermodel.build(result)
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
            hypermodel.fit(train_input, train['y'], epochs=epochs, batch_size=batch_size,
                           class_weight=class_weights,
                           callbacks=[early_stopping],
                           validation_data=(val_input, val['y']))

            val_result = hypermodel.evaluate(val_input, val['y'])
            print(f'Val result: {val_result}')
            val_results.append(val_result)
        else:
            val_results.append(model.evaluate(val_input, val['y']))
            test_score = model.evaluate(test_input, test['y'])
            test_results.append(test_score)

    # print(f'Val results: {val_results}')
    # print(f'Best hyperparameters: {best_hp}')
    print(f'Test results: {test_results}')

    # Train baseline models
    # train_ML_text_model(train_input, test_input, train['y'], test['y'])


# train_classifer_with_sentence_embedding()
train_model('large')
# baseline_LR()


graph.close()
