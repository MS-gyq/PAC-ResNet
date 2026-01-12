from torch.utils.data import random_split
from torch.utils.data import DataLoader
from data_loader import pxCountDatasetV2
from model import PAC_ResNet
from exp import run_model


seed_everything(2025)  # 设置随机种子

#================数据加载==================
train_dataset, test_dataset = dataset[:-480], dataset[-480:]

train_dataset, val_dataset = random_split(train_dataset, [int(len(train_dataset)*0.9),
                      len(train_dataset) - int(len(train_dataset)*0.9)])
val_loader = DataLoader(val_dataset, batch_size=48)
train_loader = DataLoader(train_dataset, batch_size=48)
test_loader = DataLoader(test_dataset, batch_size=48)

x_channels, x_height, x_width = x_img.shape

print("=" * 4 + "mPCResCNN_nFlows_POI_ext 自学习残差" + "=" * 4)
myModel = PAC_ResNet(in_channels=x_img.shape[0], out_channels=1, num_res_units=3, is_res=True, in_POI=x_POI.shape[0], in_ext=x_ext.shape[0], grid_size = (18,22))
rmse, mae, mape, smape, r2 = run_model(train_loader, val_loader, test_loader, m=myModel, epochs=500)
