torch 2 tflite: failure documentation
---

This is a simplified example to reproduce an issue I am having when trying to convert a model from PyTorch to TensorFlow Lite.

Model architecture: 
- `torchvision.models.detection.fasterrcnn_resnet50_fpn`: Faster R-CNN model with a ResNet-50-FPN backbone 

The same error appears with a different backbone (e.g. `fasterrcnn_mobilenet_v3_large_320_fpn`), however that one takes even longer to run, so I am not including it here.

## Setup

```
python3 -V  # 3.8.10 on my system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run conversion scripts:

```
python step1_torch2onnx.py
python step2_onnx2tf.py
python step3_tf2tflite.py
```

**Download assets** (optional): If you want to skip scripts 1 and 2 on your machine, download the prepared onnx and tensorflow models. Download and then unzip as `assets/`: https://florianletsch.de/media/torch2tflite-assets.zip (498 MB)

## What will happen

Each script will create an additional model file in the subdirectory `assets`.

The scripts have different runtimes (measured on my machine, with an i7 cpu):

1. `step1_torch2onnx.py`: 15 seconds
2. `step2_onnx2tf.py`: 912 seconds
3. `step3_tf2tflite.py`: about 18 minutes, then fails with an error

If you don't want to run them yourself, see `logs/` for the outputs on my machine. 

## Error message:

The final script `step3_tf2tflite.py` fails with this error:

```
loc(callsite(callsite("onnx_tf_prefix_If_1347@__inference___call___15681" at "StatefulPartitionedCall@__inference_signature_wrapper_16224") at "StatefulPartitionedCall")): error: could not rewrite use of immutable bound input
Traceback (most recent call last):
  File "/home/florian/projects/torch2tflite-failing/venv/lib/python3.8/site-packages/tensorflow/lite/python/convert.py", line 291, in toco_convert_protos
    model_str = wrap_toco.wrapped_toco_convert(model_flags_str,
  File "/home/florian/projects/torch2tflite-failing/venv/lib/python3.8/site-packages/tensorflow/lite/python/wrap_toco.py", line 32, in wrapped_toco_convert
    return _pywrap_toco_api.TocoConvert(
Exception: <unknown>:0: error: loc(callsite(callsite("onnx_tf_prefix_If_1347@__inference___call___15681" at "StatefulPartitionedCall@__inference_signature_wrapper_16224") at "StatefulPartitionedCall")): could not rewrite use of immutable bound input
<unknown>:0: note: loc("StatefulPartitionedCall"): called from


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "step3_tf2tflite.py", line 7, in <module>
    tflite_model = converter.convert()
  File "/home/florian/projects/torch2tflite-failing/venv/lib/python3.8/site-packages/tensorflow/lite/python/lite.py", line 913, in convert
    result = _convert_saved_model(**converter_kwargs)
  File "/home/florian/projects/torch2tflite-failing/venv/lib/python3.8/site-packages/tensorflow/lite/python/convert.py", line 722, in convert_saved_model
    data = toco_convert_protos(
  File "/home/florian/projects/torch2tflite-failing/venv/lib/python3.8/site-packages/tensorflow/lite/python/convert.py", line 297, in toco_convert_protos
    raise ConverterError(str(e))
tensorflow.lite.python.convert.ConverterError: <unknown>:0: error: loc(callsite(callsite("onnx_tf_prefix_If_1347@__inference___call___15681" at "StatefulPartitionedCall@__inference_signature_wrapper_16224") at "StatefulPartitionedCall")): could not rewrite use of immutable bound input
<unknown>:0: note: loc("StatefulPartitionedCall"): called from
```
