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

### ML Demo
```sh


```