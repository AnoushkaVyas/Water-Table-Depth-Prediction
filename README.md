# Water-Table-Depth-Prediction-Pytorch
![Python](https://img.shields.io/badge/LICENSE-MIT-blue?logo=appveyor&style=for-the-badge)
![Python](https://img.shields.io/badge/built--with-Python-green?logo=appveyor&style=for-the-badge)

### Requirements
```
Python3.x
pytorch>=0.4.0
numpy>=1.14.0
pandas>=0.22.0
scikit-learn>=0.14
```
### Installation
To use this code, please do:
 
1. Install dependencies:
    ```Shell
    pip install matplotlib numpy pandas scikit-learn
    ```
    For pytorch installation, please see in [PyTorch.org](https://pytorch.org/).
  
2. To run the code, please run:
    ```Shell
    cd Water-Table-Depth-Prediction
    python3 train.py
    ```

### Various Models available

1. LSTM
2. LSTM w/o dropout
3. Double LSTM
4. GRU
5. RNN
6. FFRNN

### Folder info
1. `checkpoints` contains saved models
2. `data` contains dataset
3. `results` contains predictions and groundtruth water table depth
4. `plots` contains all the graphs of predictions
