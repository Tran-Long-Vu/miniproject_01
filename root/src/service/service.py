# Service file for ONNX

# from root.utils.libs import *
from __future__ import annotations
import bentoml
import numpy as np

from bentoml.io import NumpyNdarray
from transformers import pipeline

# Runner
BENTO_MODEL_TAG = 'artworkclassifier:4j64hdw2i2dvcasc' # TODO: put this in configs.py later.
c_runner = bentoml.onnx.get(BENTO_MODEL_TAG).to_runner() # model in bento.yaml into runner

# Init service
artworkclassifier = bentoml.Service("artworkclassifier", runners=[c_runner])

#api specify
@artworkclassifier.api(input=NumpyNdarray(),output=NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    return c_runner.predict.run(input_data)










