from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from utils import ModelMGPU


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from model import color_net

# parameters
img_rows, img_cols = 224, 224
num_classes = 8
batch_size = 32
nb_epoch = 5

# initialise model
model = color_net(num_classes)

filepath = 'color_weights.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
            'data/train',
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)

test_set = test_datagen.flow_from_directory(
            'data/val',
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')

# model = multi_gpu_model(model, gpus=3)
# model = ModelMGPU(model, gpus=3)

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
        training_set,
        steps_per_epoch=15000,
        epochs=nb_epoch,
        validation_data=test_set,
        validation_steps=5000,
        callbacks=callbacks_list,
        use_multiprocessing=True,
        workers=8
)

model.save('color_model.h5')