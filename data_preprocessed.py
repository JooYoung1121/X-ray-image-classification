import numpy as np
import pandas as pd
from glob import glob
from sklearn.utils import shuffle
from keras.preprocessing import image
from tqdm import tqdm
import pickle

def path_to_tensor(img_path, shape):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=shape)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)/255
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, shape):
    list_of_tensors = [path_to_tensor(img_path, shape) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


df = pd.read_csv('data/Data_Entry_2017.csv')

diseases = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Nodule', 'Pneumothorax', 'Atelectasis',
            'Pleural_Thickening', 'Mass', 'Edema', 'Consolidation', 'Infiltration', 'Fibrosis', 'Pneumonia']
# Number diseases
for disease in diseases:
    df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)

# #test to perfect
# df = df.drop(df[df['Emphysema']==0][:-127].index.values)

# remove Y after age
df['Age'] = df['Patient Age'].apply(lambda x: x)
#df['Age Type'] = df['Patient Age'].apply(lambda x: x)
#df.loc[df['Age Type'] == 'M', ['Age']] = df[df['Age Type'] == 'M']['Age'].apply(lambda x: round(x / 12.))
#df.loc[df['Age Type'] == 'D', ['Age']] = df[df['Age Type'] == 'D']['Age'].apply(lambda x: round(x / 365.)).astype(int)
# remove outliers
df = df.drop(df['Age'].sort_values(ascending=False).head(1).index)
df['Age'] = df['Age'] / df['Age'].max()

# one hot data
# df = df.drop(df.index[4242])
df = df.join(pd.get_dummies(df['Patient Gender']))
df = df.join(pd.get_dummies(df['View Position']))

# random samples
df = shuffle(df)

# get other data
data = df[['Age', 'F', 'M', 'AP', 'PA']]
data = np.array(data)

labels = df[diseases].as_matrix()
files_list = ('data/images/' + df['Image Index']).tolist()

# #test to perfect
# labelB = df['Emphysema'].tolist()

labelB = (df[diseases].sum(axis=1) > 0).tolist()
labelB = np.array(labelB, dtype=int)

train_labels = labelB[:89600][:, np.newaxis]
valid_labels = labelB[89600:100800][:, np.newaxis]
test_labels = labelB[100800:][:, np.newaxis]

train_data = data[:89600]
valid_data = data[89600:100800]
test_data = data[100800:]

img_shape = (64, 64)
train_tensors = paths_to_tensor(files_list[:89600], shape = img_shape)
valid_tensors = paths_to_tensor(files_list[89600:100800], shape = img_shape)
test_tensors = paths_to_tensor(files_list[100800:], shape = img_shape)


# label = shape -> (갯수,1)
# data = shapr -> (갯수, 5) (Age,F,M,AP,PA) -> AP는 찍는 곳이 등이고 PA는 찍는곳이 가슴쪽
# tensor = image(shape) (갯수,64,64,3) 맨 뒤는 rgb -> 솔직히 가장 중요한 부분인듯


train_filename = "data_preprocessed/train_data_sample_rgb.p"
pickle.dump((train_labels, train_data, train_tensors), open(train_filename, 'wb'),protocol=4)

valid_filename = "data_preprocessed/valid_data_sample_rgb.p"
pickle.dump((valid_labels, valid_data, valid_tensors), open(valid_filename, 'wb'),protocol=4)

test_filename = "data_preprocessed/test_data_sample_rgb.p"
pickle.dump((test_labels, test_data, test_tensors), open(test_filename, 'wb'),protocol=4)
