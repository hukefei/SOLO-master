# import onnxruntime
# from PIL import Image
# from mmdet.datasets.pipelines.transforms import Resize
#
#
# session = onnxruntime.InferenceSession("solo.onnx")
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
# input_shape = session.get_inputs()[0].shape
#
# trans = Resize(img_scale=(832, 512), keep_ratio=False)
#
# image = Image.open('../test.jpg')
# image = trans(image).unsqueeze(0).numpy()
# res = session.run([output_name], {input_name: image})
# print(res)

from onnx_coreml import convert

# Load the ONNX model as a CoreML model
model = convert(model='../../solo2.onnx', )

# Save the CoreML model
model.save('../../solo2.mlmodel')