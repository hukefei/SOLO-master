
from onnx_coreml import convert

# Load the ONNX model as a CoreML model
model = convert(model='../../solo2.onnx', minimum_ios_deployment_target='13')

# Save the CoreML model
model.save('../../solo2.mlmodel')