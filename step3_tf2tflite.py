import time

import tensorflow as tf

t_start = time.time()
converter = tf.lite.TFLiteConverter.from_saved_model('assets/tfsavedmodel')

# needed to convert mutable variables
converter.experimental_enable_resource_variables = True  # requires tf-nightly or tf-nightly-cpu

# needed to support tf.CropAndResize and tf.Range
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()
with open('assets/model.tflite', 'wb') as f:
    f.write(tflite_model)

duration = int(time.time() - t_start)
print(f"Done. Time elapsed: {duration} seconds")
