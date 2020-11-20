
import tensorflow as tf

import pandas as pd
from tensorflow.keras.models import Sequential, load_model

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split


oval = pd.read_csv('oval_capsule/OVAL.csv')
capsule = pd.read_csv('oval_capsule/CAPSULE.csv')

oval['is_oval'] = 'oval'
oval['splimage_path'] = 'oval_capsule/OVAL/' + oval['splimage'] + '.jpg'
capsule['is_oval'] = 'not_oval'
capsule['splimage_path'] = 'oval_capsule/CAPSULE/' + capsule['splimage'] + '.jpg'
df = pd.concat([oval, capsule], ignore_index=True)

df = df[['ID', 'splshape_text', 'splimage', 'is_oval', 'splimage_path']]

Img_size = (256, 256)
train_df, test_df = train_test_split(df, test_size=0.3,
                                     random_state=1,
                                     stratify=df[['is_oval']])
train_dfe, val_df = train_test_split(train_df, random_state=1, test_size=0.2, stratify=train_df[['is_oval']])


def predict_oval(x):
    per = x[0] * 100
    if per >= 50.0:
        return 'OVAL'
    else:
        return 'CAPSULE'


model = load_model('ai_model2/')

# img = load_img('oval_capsule/CAPSULE/1pt5mg_Capsule.jpg', target_size=(256, 256))
for index, row in test_df.iterrows():
    img_path = row.splimage_path
    img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(img) * (1. / 255.)
    img_array = tf.expand_dims(img_array, 0)
    predictor = model.predict(img_array)
    predict = predict_oval(predictor)
    test_df.loc[index, 'Predict_shape'] = predict

test_df.to_csv('test_df2.csv')
