import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

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

TrainImageDataGenerator = ImageDataGenerator(
    height_shift_range=0.1,
    width_shift_range=0.1,
    rotation_range=5,
    shear_range=0.01,
    fill_mode='reflect',
    zoom_range=0.15,
    rescale=1. / 255.
)
testImageGenerator = ImageDataGenerator(
    # rescale=1. / 255.
)

train_gen = TrainImageDataGenerator.flow_from_dataframe(
    train_dfe, x_col='splimage_path', y_col='is_oval', class_mode='binary', batch_size=28,
    target_size=Img_size
)
val_gen = TrainImageDataGenerator.flow_from_dataframe(
    val_df, x_col='splimage_path', y_col='is_oval', class_mode='binary', batch_size=28,
    target_size=Img_size
)
test_gen = testImageGenerator.flow_from_dataframe(
    test_df, x_col='splimage_path', y_col='is_oval', class_mode='binary', batch_size=28, shuffle=False,
    target_size=Img_size,
)

model = Sequential([

    layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPool2D(pool_size=(2, 2)),

    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')

])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
)
model.save('ai_model2/')
