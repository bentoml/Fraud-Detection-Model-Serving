<div align="center">
    <h1 align="center">Fraud Detection Models Serving</h1>
    <i>Powered by BentoML 🍱</i>
    <br>
</div>
<br>

## 📖 Introduction 📖
This project demonstrates how to serve a fraud detection model trained with [XGBoost]() on the 
dataset from the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data).


## 🏃‍♂️ Getting Started 🏃‍♂️

### 0. Clone the Repository:

```bash
git clone git@github.com:bentoml/Fraud-Detection-Model-Serving.git
cd Fraud-Detection-Model-Serving
```

### 1. Install Dependencies:
```bash
pip install -r ./dev-requirements.txt
```

### 2. Download dataset

Before downloading, set up your Kaggle API Credentials following instructions 
[here](https://github.com/Kaggle/kaggle-api#api-credentials) and accept the [dataset 
rules on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

```bash
./download_data.sh
```

### 3. Model Training

Execute the `./IEEE-CIS-Fraud-Detection.ipynb` notebook with the `train.sh` script:
```bash
./train.sh
```

This will create 3 variations of the model, you can view and manage those models via the 
`bentoml models`  CLI commnad:

```bash
$ bentoml models list

Tag                                         Module           Size        Creation Time
ieee-fraud-detection-tiny:qli6n3f6jcta3uqj  bentoml.xgboost  141.40 KiB  2023-03-08 23:03:36
ieee-fraud-detection-lg:o7wqb5f6jcta3uqj    bentoml.xgboost  18.07 MiB   2023-03-08 23:03:17
ieee-fraud-detection-sm:5yblgmf6i2ta3uqj    bentoml.xgboost  723.00 KiB  2023-03-08 22:52:16
```

Saved models can also be accessed via the BentoML Python API, e.g.:

```python
import bentoml
import pandas as pd
import numpy as np

model_ref = bentoml.xgboost.get("ieee-fraud-detection-sm:latest")
model_runner = model_ref.to_runner()
model_runner.init_local()
model_preprocessor = model_ref.custom_objects["preprocessor"]

test_transactions = pd.read_csv("./data/test_transaction.csv")[0:500]
test_transactions = model_preprocessor.transform(test_transactions)
result = model_runner.predict_proba.run(test_transactions)
print(np.argmax(result, axis=1))
```


### 4. Serving the model

The `service.py` file contains the source code for defining an ML service:

```python
import numpy as np
import pandas as pd
from sample import sample_input

import bentoml
from bentoml.io import JSON
from bentoml.io import PandasDataFrame

model_ref = bentoml.xgboost.get("ieee-fraud-detection-lg:latest")
preprocessor = model_ref.custom_objects["preprocessor"]
fraud_model_runner = model_ref.to_runner()

svc = bentoml.Service("fraud_detection", runners=[fraud_model_runner])

input_spec = PandasDataFrame.from_sample(sample_input)

@svc.api(input=input_spec, output=JSON())
async def is_fraud(input_df: pd.DataFrame):
    input_df = input_df.astype(sample_input.dtypes)
    input_features = preprocessor.transform(input_df)
    results = await fraud_model_runner.predict_proba.async_run(input_features)
    predictions = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    return {"is_fraud": list(map(bool, predictions)), "is_fraud_prob": results[:, 1]}
```


Run `bentoml serve` command to launch the server locally:

```bash
bentoml serve
```

## 🌐 Interacting with the Service 🌐
The default mode of BentoML's model serving is via HTTP server. Here, we showcase a few examples of how one can interact with the service:
### Swagger UI
Visit `http://localhost:3000/` in a browser and send test requests via the UI.

### cURL
Via the command `curl`, you can:
```bash
head --lines=200 ./data/test_transaction.csv | curl -X POST -H 'Content-Type: text/csv' --data-binary @- http://0.0.0.0:3000/is_fraud
```

### Via BentoClient 🐍
```python
import pandas as pd
from bentoml.client import Client

test_transactions = pd.read_csv("./data/test_transaction.csv")[0:500]
client = Client.from_url('localhost:3000')

results = client.is_fraud(test_transaction)
print(results)
```


## 🚀 Bringing it to Production 🚀
BentoML offers a number of options for deploying and hosting online ML servicesinto
for production, learn more at [Deploying Bento Docs](https://docs.bentoml.org/en/latest/concepts/deploy.html).

---

In this README, we will go over a basic deployment strategy with Docker containers.


### 1. Build a docker image

Build a Bento to lock the model version and dependency tree:
```bash
bentoml build
```

Ensure docker is installed and running, build a docker image with `bentoml containerize`
```bash
bentoml containerize fraud_detection:latest
```

Test out the docker image built:

```bash
docker run -it --rm -p 3000:3000 fraud_detection:{YOUR BENTO VERSION}
```

### 2. Inference on GPU

Use `bentofile-gpu.yaml` to build a new Bento, which adds the following two lines to the YAML.
This ensures the docker image comes with GPU libraries installed and BentoML will automatically
load models on GPU when running the docker image with GPU devices available.

```yaml
docker:
  cuda_version: "11.6.2"
```

Build Bento with GPU support:
```bash
bentoml build -f ./bentofile-gpu.yaml
```

Build and run docker image with GPU enabled:
```bash
bentoml containerize fraud_detection:latest

docker run --gpus all --device /dev/nvidia0 \
           --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
           --device /dev/nvidia-modeset --device /dev/nvidiactl \
           fraud_detection:{YOUR BENTO VERSION}
```

### 3. Multi-model Inference Pipeline/Graph

BentoML makes it efficient to create ML service with multiple ML models, which is often used for combining
multiple fraud detection models and getting an aggregated result. With BentoML, users can choose to run
models sequentially or in parallel using the Python AsyncIO APIs along with Runners APIs. This makes
it possible create inference graphes or multi-stage inference pipeline all from Python APIs.

An example can be found under `inference_graph_demo` that runs all three models simutaneously and 
aggregate their results:

```bash
cd inference_graph_demo

bentoml serve
```

Learn more about BentoML Runner usage [here](https://docs.bentoml.org/en/latest/concepts/runner.html)


### 4. Benchmark Testing

Visit the `/benchmark/README.md` for how to run benchmark tests on your fraud detection service and 
understanding its throughput and latency on your deployment target.


## 👥Join our Community 👥

BentoML has a thriving open source community where thousands of ML/AI practitioners are contributing to the project, helping other users and discussing the future of AI. [👉 Join us on slack today!](https://l.bentoml.com/join-slack)
