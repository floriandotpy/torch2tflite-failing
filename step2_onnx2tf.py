import time

import onnx
from onnx_tf.backend import prepare

t_start = time.time()
print("Loading...")
onnx_model = onnx.load('assets/model.onnx')
print("Writing onnx debug files")
with open('assets/onnx-full.txt', 'w') as fp:
    fp.write(str(onnx_model))
with open('assets/onnx.txt', 'w') as fp:
    fp.write(str(onnx.helper.printable_graph(onnx_model.graph)))
print("Converting...")
model_tf = prepare(onnx_model)
print("Exporting...")
model_tf.export_graph('assets/tfsavedmodel')
duration = int(time.time() - t_start)
print(f"Done. Time elapsed: {duration} seconds")
