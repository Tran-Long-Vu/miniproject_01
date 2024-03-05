# Service file for ONNX

#from root.utils.libs import * 
from __future__ import annotations
import bentoml
import numpy as np
from PIL import Image
from bentoml.io import Image
from transformers import pipeline
import onnx 

# save onnx in bento models list
onnx_model = onnx.load("./src/ArtworkClassifier.onnx") 
signatures = {
    "run": {"batchable": True},
}
bentoml.onnx.save_model("ArtworkClassifier",
                        onnx_model,
                        signatures=signatures) 
runner = bentoml.onnx.get("ArtworkClassifier:latest").to_runner() # model in bento.yaml

# Define the service
#@bentoml.env(pip_packages=["onnxruntime", "Pillow"])
#@bentoml.artifacts([("model", bentoml.artifacts.OnnxModelArtifact)])

# TODO: Change output to string later
class ArtworkClassifierService(bentoml.Service):
    def __init__(self) -> None:
        self.pipeline = pipeline('image-classification') #use pipeline
    
    # Model API Inference
    @bentoml.api(input_spec=Image(), output_spec=Image(pilmode="L"))
    def sr(self, img):
        ort_session = bentoml.onnx.load_model("artworkclassifier")
        img = img.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        #output
        sr_arr = ort_session.run([output_name], 
        {input_name: arr.astype(np.float32)})
        sr_arr = np.squeeze(sr_arr)
        sr_arr = np.uint8(sr_arr * 255) #string conv
        return sr_arr

# Init service
svc = ArtworkClassifierService()


