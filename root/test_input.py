#input test
from utils.libs import *

test_input = np.random.randn(1, 3, 224, 224)
providers=["CPUExecutionProvider"]
bento_model = bentoml.onnx.get("artworkclassifier")

runner = bento_model.with_options(providers=providers).to_runner()

runner.init_local() # remove when deploy

runner.run.run(test_input)
print(runner.run.run(test_input))
