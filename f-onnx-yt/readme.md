## [ONNX](https://github.com/onnx/onnx/tree/main)

### Why Onnx?
* Intermediary ML framework used to convert between different ML frameworks
* Lets say Torch to tflite ; or H5 to Tensorrt

### Challenges in DL
* Inference is slow from the core exported models.
* Core models are heavy and unoptimized.
* Quantization through ONNX helps it make lighter and increase throughput.
* Need to have some harware acceleration. Not possible in models from core frameworks.

### Design Principles
* Supports DNN but also Traditional ML
* Very Flexible
* Standardized list of well defined operations.

### File format
* Model -> Version info ; Meta Data ; Acyclic computation dataflow graph
* Graph -> Inputs and outputs ; List of comp nodes ; Graph name
* Comp Node -> Zero or more inputs of defined types ; One or more O/p ; Operator & its parameters

### Data Types
* Tensor type -> int8/16/32/64 ; uint8/16/32/64 ; float16, double, double ; bool; string ; complex64/128
* Non Tensor -> Sequence, Map (For ML)

### ML Demo (Visualize the models on [netron](https://netron.app/))
```py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X,y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
joblib.dump(clf, 'models/random_forest.pkl')

```

```py
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

cls = joblib.load('./models/random_forest.pkl')
initial_type =[('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(cls, initial_types=initial_type)

with open('./models/model.onnx' , 'wb') as output:
    output.write(onnx_model.SerializeToString())
```

```py
import onnxruntime as rt
import numpy as np

data = np.array([[4.5,4.9,5.1,5.4], [4.7,4.1,5.1,5.1], [1.1,1.1,1.3,5.9]])
session = rt.InferenceSession("./models/model.onnx")

ip = session.get_inputs()[0].name
op = session.get_outputs()[0].name

preds = session.run([op], {ip : data.astype(np.float32)})[0]
preds
```