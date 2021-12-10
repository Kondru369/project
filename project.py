# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:19:47.463192Z","iopub.execute_input":"2021-12-10T08:19:47.463646Z","iopub.status.idle":"2021-12-10T08:19:54.101168Z","shell.execute_reply.started":"2021-12-10T08:19:47.463560Z","shell.execute_reply":"2021-12-10T08:19:54.100121Z"}}
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import os
import numpy as np
import pandas as np

import matplotlib.pyplot as plt
%matplotlib inline


# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:23:42.274140Z","iopub.execute_input":"2021-12-10T08:23:42.274474Z","iopub.status.idle":"2021-12-10T08:23:43.100751Z","shell.execute_reply.started":"2021-12-10T08:23:42.274443Z","shell.execute_reply":"2021-12-10T08:23:43.100040Z"}}

img_normal = load_img('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0115-0001.jpeg' )

print('NORMAL')
plt.imshow(img_normal)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:24:26.296790Z","iopub.execute_input":"2021-12-10T08:24:26.297121Z","iopub.status.idle":"2021-12-10T08:24:26.644463Z","shell.execute_reply.started":"2021-12-10T08:24:26.297088Z","shell.execute_reply":"2021-12-10T08:24:26.643213Z"}}

img_pneumonia = load_img('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1000_bacteria_2931.jpeg' )

print('PNEUMONIA')
plt.imshow(img_pneumonia)
plt.show()

# %% [markdown]
# 

# %% [markdown]
# **Defining the image width and height**

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:25:11.365973Z","iopub.execute_input":"2021-12-10T08:25:11.366325Z","iopub.status.idle":"2021-12-10T08:25:11.371719Z","shell.execute_reply.started":"2021-12-10T08:25:11.366286Z","shell.execute_reply":"2021-12-10T08:25:11.370348Z"}}
# dimensions of our images.
img_width, img_height = 150, 150

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:26:34.949434Z","iopub.execute_input":"2021-12-10T08:26:34.949702Z","iopub.status.idle":"2021-12-10T08:26:34.955084Z","shell.execute_reply.started":"2021-12-10T08:26:34.949675Z","shell.execute_reply":"2021-12-10T08:26:34.954167Z"}}
train_data_dir = '../input/chest-xray-pneumonia/chest_xray/train'
validation_data_dir = '../input/chest-xray-pneumonia/chest_xray/val'
test_data_dir = '../input/chest-xray-pneumonia/chest_xray/test'

nb_train_samples = 5217
nb_validation_samples = 17
epochs = 20
batch_size = 16

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:26:46.996634Z","iopub.execute_input":"2021-12-10T08:26:46.996945Z","iopub.status.idle":"2021-12-10T08:26:47.001816Z","shell.execute_reply.started":"2021-12-10T08:26:46.996901Z","shell.execute_reply":"2021-12-10T08:26:47.001012Z"}}
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# %% [markdown]
# ****

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:29:56.424043Z","iopub.execute_input":"2021-12-10T08:29:56.425223Z","iopub.status.idle":"2021-12-10T08:29:56.521153Z","shell.execute_reply.started":"2021-12-10T08:29:56.425166Z","shell.execute_reply":"2021-12-10T08:29:56.520445Z"}}
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:30:56.207226Z","iopub.execute_input":"2021-12-10T08:30:56.207720Z","iopub.status.idle":"2021-12-10T08:30:56.216466Z","shell.execute_reply.started":"2021-12-10T08:30:56.207688Z","shell.execute_reply":"2021-12-10T08:30:56.215678Z"}}
model.layers

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:31:10.358306Z","iopub.execute_input":"2021-12-10T08:31:10.358716Z","iopub.status.idle":"2021-12-10T08:31:10.364015Z","shell.execute_reply.started":"2021-12-10T08:31:10.358687Z","shell.execute_reply":"2021-12-10T08:31:10.363125Z"}}
model.input

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:31:19.966293Z","iopub.execute_input":"2021-12-10T08:31:19.966684Z","iopub.status.idle":"2021-12-10T08:31:19.974837Z","shell.execute_reply.started":"2021-12-10T08:31:19.966655Z","shell.execute_reply":"2021-12-10T08:31:19.974093Z"}}
model.output

# %% [markdown]
# **Compile**

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:32:16.171888Z","iopub.execute_input":"2021-12-10T08:32:16.172219Z","iopub.status.idle":"2021-12-10T08:32:16.191912Z","shell.execute_reply.started":"2021-12-10T08:32:16.172190Z","shell.execute_reply":"2021-12-10T08:32:16.190950Z"}}
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:32:28.022000Z","iopub.execute_input":"2021-12-10T08:32:28.022321Z","iopub.status.idle":"2021-12-10T08:32:28.028406Z","shell.execute_reply.started":"2021-12-10T08:32:28.022292Z","shell.execute_reply":"2021-12-10T08:32:28.026912Z"}}
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:32:40.020255Z","iopub.execute_input":"2021-12-10T08:32:40.021467Z","iopub.status.idle":"2021-12-10T08:32:40.026644Z","shell.execute_reply.started":"2021-12-10T08:32:40.021393Z","shell.execute_reply":"2021-12-10T08:32:40.025670Z"}}
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:32:53.511907Z","iopub.execute_input":"2021-12-10T08:32:53.512250Z","iopub.status.idle":"2021-12-10T08:32:57.313824Z","shell.execute_reply.started":"2021-12-10T08:32:53.512217Z","shell.execute_reply":"2021-12-10T08:32:57.312010Z"}}
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:33:09.618424Z","iopub.execute_input":"2021-12-10T08:33:09.618750Z","iopub.status.idle":"2021-12-10T08:33:09.733623Z","shell.execute_reply.started":"2021-12-10T08:33:09.618719Z","shell.execute_reply":"2021-12-10T08:33:09.732988Z"}}
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:33:20.342115Z","iopub.execute_input":"2021-12-10T08:33:20.342807Z","iopub.status.idle":"2021-12-10T08:33:20.458915Z","shell.execute_reply.started":"2021-12-10T08:33:20.342768Z","shell.execute_reply":"2021-12-10T08:33:20.457840Z"}}
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:33:40.246640Z","iopub.execute_input":"2021-12-10T08:33:40.247006Z","iopub.status.idle":"2021-12-10T08:33:40.569281Z","shell.execute_reply.started":"2021-12-10T08:33:40.246960Z","shell.execute_reply":"2021-12-10T08:33:40.567987Z"}}
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T08:34:08.158366Z","iopub.execute_input":"2021-12-10T08:34:08.158679Z","iopub.status.idle":"2021-12-10T09:20:32.876615Z","shell.execute_reply.started":"2021-12-10T08:34:08.158650Z","shell.execute_reply":"2021-12-10T09:20:32.875381Z"}}
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T09:21:30.840279Z","iopub.execute_input":"2021-12-10T09:21:30.840619Z","iopub.status.idle":"2021-12-10T09:21:30.898815Z","shell.execute_reply.started":"2021-12-10T09:21:30.840585Z","shell.execute_reply":"2021-12-10T09:21:30.898115Z"}}
model.save('my_model.h5')

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2021-12-10T09:22:39.220668Z","iopub.execute_input":"2021-12-10T09:22:39.221176Z","iopub.status.idle":"2021-12-10T09:22:50.216357Z","shell.execute_reply.started":"2021-12-10T09:22:39.221142Z","shell.execute_reply":"2021-12-10T09:22:50.215427Z"}}
scores = model.evaluate_generator(test_generator)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
