# PAC-ResNet
The core implementation of PAC-ResNet

## Environment

The code was developed and tested with the following environment:

- Python 3.8
- PyTorch 2.3.1 (CUDA 11.8, cuDNN 8.7.0)
- CUDA Toolkit 11.8
- pandas 2.0.3
- numpy 1.24.3

## ▶️ Run the Code

To run the main script:

```bash
python run.py
```
## 📂 Data

The original dataset contains sensitive information and is not publicly available due to privacy and legal restrictions.  

> 🔒 We are unable to share the raw or processed data upon request.

### 🧩 Custom Data Integration

Since the original dataset is not publicly available, **you are encouraged to plug in your own data** by implementing a compatible `torch.utils.data.Dataset` object.

Your custom dataset should support indexing (e.g., `dataset[i]`) and return a dictionary with the following four keys:

| Key       | Description                     | Expected Shape        |
|-----------|----------------------------------|------------------------|
| `'y'`     | Target (future) tensor          | `(t, H, W)`            |
| `'x_POI'` | POI (Point-of-Interest) features| `(c_poi, H, W)`        |
| `'x_ext'` | External (e.g., weather) features| `(c_ext, 1, 1)`       |
| `'x_img'` | Historical input sequence       | `(k, H, W)`            |

**Notation**:  
- `t`: number of future time steps to predict  
- `k`: number of historical time steps as input  
- `c_poi`: number of POI categories  
- `c_ext`: number of external (environmental) covariates  
- `H`, `W`: spatial height and width (e.g., grid dimensions)

> ✅ All tensors must be of type `torch.Tensor`.  
> ✅ For each sample, `H` and `W` must be consistent across all four tensors.

#### Example Dataset Implementation
```python
from torch.utils.data import Dataset
import torch

class MyCustomDataset(Dataset):
    def __init__(self):
        # Initialize your data paths or metadata here
        pass

    def __len__(self):
        return 100  # Replace with actual number of samples

    def __getitem__(self, idx):
        # Replace with your actual data loading logic
        y = torch.randn(t, H, W)           # shape: (t, H, W)
        x_poi = torch.randn(c_poi, H, W)   # shape: (c_poi, H, W)
        x_ext = torch.randn(c_ext, 1, 1)   # shape: (c_ext, 1, 1)
        x_img = torch.randn(k, H, W)       # shape: (k, H, W)

        return {
            'y': y,
            'x_POI': x_poi,
            'x_ext': x_ext,
            'x_img': x_img
        }
```


