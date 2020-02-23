# mol_mpnn
Tensorflow implementation for efficient learning of message passing neural networks for molecular property production

## Components
- **preprocessing.py** - script for preprocessing data
- **train.py** - script for model training
- **test.py** - script for model evaluation (molecular property prediction)
- **MPNN.py** - model architecture

## Dependencies
- **Python**
- **TensorFlow**
- **RDKit**
- **NumPy**
- **scikit-learn**
- **sparse**

## Results (Mean Absolute Error on Test Set)
| Property | MAE    |
|----------|--------|
| mu       | 0.0293 |
| alpha    | 0.0131 |
| HOMO     | 0.0557 |
| LUMO     | 0.0271 |
| gap      | 0.0395 |
| R2       | 0.0049 |
| ZPVE     | 0.0029 |
| U0       | 0.0013 |
| U298     | 0.0013 |
| H298     | 0.0015 |
| G298     | 0.0013 |
| Cv       | 0.0117 |
