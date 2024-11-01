import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
import pandas as pd
import concurrent.futures
import os
from glob import glob
import time  # For tracking time

### Path setup
dcmex_2ds = '/gws/nopw/j04/dcmex/users/ezriab/processed_images/2ds/ch_0/220730153000/'
file_path = dcmex_2ds
file_list = glob(file_path + '*.png')
file_names = [os.path.basename(file_path) for file_path in file_list]

save_loc = '/gws/nopw/j04/dcmex/users/ezriab/processed_images/2ds/ch_0/220730153000/'
save_name = 'habit_predictions.csv'

categories = ['CA', 'Co', 'CC', 'CBC', 'CG', 'HPC', 'Dif', 'FA', 'WD']
columns_names = ['Name', 'Category'] + categories

### Model setup with custom layer
model_path = '/gws/nopw/j04/dcmex/users/ezriab/cp-of-Jaffeux-22/novmodel.h5py'

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

custom_objects = {'RandomFlip': RandomFlip}
model = load_model(model_path, custom_objects=custom_objects)

### Helper functions
def load_image_with_timeout(file_path, target_size=(200, 200), color_mode='grayscale', timeout=15):
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(image.load_img, file_path, target_size=target_size, color_mode=color_mode)
            img = future.result(timeout=timeout)
        return img
    except concurrent.futures.TimeoutError:
        print(f"Timeout loading {file_path}. Skipping.")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None

### Processing loop
results = []

with open(f'{save_loc}cnn_run_output.txt', "w") as file:
    total_start_time = time.time()  # Track overall start time
    for i in range(len(file_names)):
        start_time = time.time()  # Track start time for each image
        file.write(f'{file_names[i]} start\n')
        print(f'Processing {file_names[i]}')
        try:
            # Load image with timeout handling
            img = load_image_with_timeout(file_path + file_names[i])
            if img is None:
                file.write(f'{file_names[i]} skipped: load issue\n')
                continue

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Check for any unexpected shapes or sizes
            if img_array.shape != (1, 200, 200, 1):  # Expected shape for (batch, height, width, channels)
                file.write(f'{file_names[i]} skipped: unexpected shape {img_array.shape}\n')
                continue

            # Predict and log results
            predictions = model.predict(img_array, verbose=0)[0]
            predicted_index = np.argmax(predictions)
            predicted_category = categories[predicted_index]

            data = {'Name': file_names[i][:-4], 'Category': predicted_category}
            for j, category in enumerate(categories):
                data[category] = predictions[j]
            
            results.append(data)
            file.write(f'{file_names[i]} completed successfully\n')

        except Exception as e:
            print(f"Error processing {file_names[i]}: {e}")
            file.write(f'Error processing {file_names[i]}: {e}\n')
        
        # Log the time taken for each image
        end_time = time.time()
        elapsed_time = end_time - start_time
        file.write(f'{file_names[i]} processing time: {elapsed_time:.2f} seconds\n')

    # Log the total processing time
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    file.write(f'Total processing time: {total_elapsed_time:.2f} seconds\n')
    print(f'Total processing time: {total_elapsed_time:.2f} seconds')

# Save to DataFrame and CSV
df = pd.DataFrame(results, columns=columns_names)
df.to_csv(save_loc + save_name, index=False)
print('Processing finished.')
