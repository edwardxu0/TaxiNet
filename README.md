# TaxiNet
The TaxiNet artifact for DNN Verification

## Usage
### Setup
1. Clone me.
2. Download the [SoI artifact](https://zenodo.org/record/6109456/) and unzip to `./lib/SoI`.
3. Create a virtual environment using `requirements.txt`
### Usage
1. Get help:
    >`./tools/convert_taxinet.py --help`
2. Convert TaxiNet32x16 from TF graph to ONNX:
    >`./tools/convert_taxinet.py soi --save_model`
3. Train your own TaxiNet32x16:
    >`./tools/convert_taxinet.py train`
4. Generate property for DNNV:
    >`./tools/convert_taxinet.py gen_prop`
### Reference
1. [Validation of Image-Based Neural Network Controllers through Adaptive Stress Testing](https://arxiv.org/abs/2003.02381)
2. [Efficient Neural Network Analysis with Sum-of-Infeasibilities](https://arxiv.org/abs/2203.11201)