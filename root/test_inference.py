#local inference testing
from utils.libs import *

#define session
ort_session = bentoml.onnx.load_model("artworkclassifier")
test_input = np.random.randn(1, 3, 224, 224)
#inference
output = ort_session.run(None, {"modelInput": test_input.astype(np.float32)})
print(output)



