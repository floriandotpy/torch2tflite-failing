import time

import tensorflow as tf

t_start = time.time()
converter = tf.lite.TFLiteConverter.from_saved_model('assets/tfsavedmodel')
tflite_model = converter.convert()
with open('assets/model.tflite', 'wb') as f:
    f.write(tflite_model)

duration = int(time.time() - t_start)
print(f"Done. Time elapsed: {duration} seconds")
