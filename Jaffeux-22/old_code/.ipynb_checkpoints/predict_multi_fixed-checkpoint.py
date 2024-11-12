import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
import csv
from glob import glob
import os
import pandas as pd
import concurrent.futures

### path stuff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dcmex_2ds = '/gws/nopw/j04/dcmex/users/ezriab/processed_images/2ds/ch_0/220730153000/'

file_path = dcmex_2ds # path to all files

# get all files in specif folder
file_list = glob(file_path+'*.png') # whole string of path + filenames in location
file_names = [os.path.basename(file_path) for file_path in file_list] # just file names


## where would you like predictions saved?
#save_loc = '/gws/nopw/j04/dcmex/users/ezriab/2dprocessed/flight_220730153000/'
save_loc = '/gws/nopw/j04/dcmex/users/ezriab/processed_images/2ds/ch_0/220730153000/'
save_name = 'habit_predictions.csv'

## quite important
categories = ['CA', 'Co',  'CC', 'CBC', 'CG', 'HPC', 'Dif', 'FA', 'WD']
columns_names = ['Name', 'Category'] + categories
######## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#### start: model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model_path = '/gws/nopw/j04/dcmex/users/ezriab/cp-of-Jaffeux-22/novmodel.h5py' # to h5 folder

# Define custom layer with serialization support
# (model would not open without this)
@register_keras_serializable()
class RandomFlip(tf.keras.layers.Layer):
    def __init__(self, mode='horizontal_and_vertical', seed=None, **kwargs):
        super(RandomFlip, self).__init__(**kwargs)
        self.mode = mode
        self.seed = seed

    def call(self, inputs, training=None):
        if training:
            if self.mode == 'horizontal_and_vertical':
                return tf.image.random_flip_left_right(tf.image.random_flip_up_down(inputs), seed=self.seed)
            elif self.mode == 'horizontal':
                return tf.image.random_flip_left_right(inputs, seed=self.seed)
            elif self.mode == 'vertical':
                return tf.image.random_flip_up_down(inputs, seed=self.seed)
        return inputs

    def get_config(self):
        config = super(RandomFlip, self).get_config()
        config.update({'mode': self.mode, 'seed': self.seed})
        return config

# Load the model with custom objects
custom_objects = {'RandomFlip': RandomFlip}
model = load_model(model_path, custom_objects=custom_objects)
#### end: model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#### start: running model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# List to store results
## something is going wrong on new images? so have try + except + new function

# Define a function to load an image
def load_image_with_timeout(file_path, target_size=(200, 200), color_mode='grayscale', timeout=15):
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(image.load_img, file_path, target_size=target_size, color_mode=color_mode)
            img = future.result(timeout=timeout)  # Set a timeout in seconds
        return img
    except concurrent.futures.TimeoutError:
        #print(f"Skipping {file_path}: loading timed out.")
        #file.write(f'Skipping {file_path}: loading timed out. \n')
        print('problem')
    except Exception as e:
        #file.write(f'Skipping {file_path}: : error - {e}\n')
        print(f"Skipping {file_path}: error - {e}")
    return None


results = []

for i in range(len(file_names)):
    with open(f'{save_loc}cnn_run_output.txt', "w") as file:
        try:
            # Load the image
            file.write(f'{file_names[i]} start \n')
            print(f'{file_names[i]}')
        
            # Attempt to load the image with a timeout
            img = load_image_with_timeout(file_path + file_names[i])
            if img is None:
                file.write(f'Skipping {file_path}: a problem occured \n')
                continue  # Skip this image if it couldn't be loaded
                
            '''#img = image.load_img(file_path + file_names[i], target_size=(200, 200), color_mode='grayscale')'''
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
            # Predict using the loaded model
            predictions = model.predict(img_array)[0]  # Set verbose=0 to reduce output
            predicted_index = np.argmax(predictions)
            predicted_category = categories[predicted_index]
    
            # Prepare the data to be saved
            data = {'Name': file_names[i][:-4], 'Category': predicted_category}
            for j, category in enumerate(categories):
                data[category] = predictions[j]
    
            # Append the data to results list
            results.append(data)
    
        except Exception as e:
            print(f"Error processing {file_names[i]}: {e}")
    
# Save the results to a DataFrame and CSV
df = pd.DataFrame(results, columns=columns_names)
df.to_csv(save_loc + save_name, index=False)
print('Finished')
#file.write(f'all images compleate\n')

#### end: running model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
