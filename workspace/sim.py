import onnx
import onnxsim
model_onnx = onnx.load("yolov8n-seg-dynamic.transd.onnx")  # load onnx model
model_onnx, check = onnxsim.simplify(model_onnx)
onnx.save(model_onnx, "yolov8n-seg-dynamic.transd.sim.onnx")
