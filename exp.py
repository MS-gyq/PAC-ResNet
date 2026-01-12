from sklearn.metrics import r2_score
from torch import optim, nn
import numpy as np
import torch
import pickle
import pandas as pd


def calculate_mape(y_true, y_pred, epsilon=1):
    """
    计算 Mean Absolute Percentage Error (MAPE)
    """
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

def calculate_smape(y_true, y_pred, epsilon=1e-10):
    """
    计算 Symmetric Mean Absolute Percentage Error (SMAPE)
    """
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator_safe = np.where(denominator < epsilon, epsilon, denominator)
    return 2 * np.mean(np.abs(y_true - y_pred) / denominator_safe) * 100

def run_model(train_loader, val_loader, test_loader, m=None, epochs=20, save_path="best_model.pt", DEV="cuda"):
    m1 = m.to(DEV)
    criterion = nn.HuberLoss()
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()
    optimizer = optim.Adam(m1.parameters(), lr=0.003)
    best_val_loss = float("inf")
    train_loss_log = []
    val_loss_log = []
    print("Begin Training")
    for epoch in range(epochs):
        m1.train()
        train_losses = []
        for batch in train_loader:
            y, x_POI, x_ext, x_lastExt, x_img = batch.values()
            optimizer.zero_grad()
            pred = m1(x_img.float().to(DEV),
                      x_POI.float().to(DEV) if x_POI is not None else None,
                      x_ext.float().to(DEV) if x_ext is not None else None)
            loss = criterion(pred, y.float().to(DEV))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ====== 验证阶段 ======
        m1.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                y, x_POI, x_ext, x_lastExt, x_img = batch.values()
                pred = m1(x_img.float().to(DEV),
                          x_POI.float().to(DEV) if x_POI is not None else None,
                          x_ext.float().to(DEV) if x_ext is not None else None)
                val_loss = criterion(pred, y.float().to(DEV)).item()
                val_losses.append(val_loss)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        train_loss_log.append(avg_train_loss)
        val_loss_log.append(avg_val_loss)

        print(f'Epoch {epoch + 1} | Train RMSE: {avg_train_loss:.4f} | Val RMSE: {avg_val_loss:.4f} | Time: {pd.Timestamp.now()}')

        # ====== 保存最优模型 ======
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = m1.state_dict()
            torch.save(best_model_state, save_path)
            print(f"Saved new best model at epoch {epoch + 1}")

    # ============== 最终加载最优模型 ==============
    m1.load_state_dict(torch.load(save_path))

    # ============== 测试阶段 ==============
    print("Begin Testing")
    m1.eval()
    res, eval_mse, eval_mae, eval_mape, eval_smape, eval_r2 = [], [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            y, x_POI, x_ext, x_lastExt, x_img = batch.values()
            pred = m1(x_img.float().to(DEV),
                      x_POI.float().to(DEV) if x_POI is not None else None,
                      x_ext.float().to(DEV) if x_ext is not None else None)
            # 计算 RMSE 和 MAE
            eval_mse.append(loss_mse(pred, y.float().to(DEV)).item())
            eval_mae.append(loss_mae(pred, y.float().to(DEV)).item())

            # 计算 MAPE 和 SMAPE
            eval_mape.append(calculate_mape(y.cpu().numpy(), pred.cpu().numpy()))
            eval_smape.append(calculate_smape(y.cpu().numpy(), pred.cpu().numpy()))

            # 计算 R² (flatten y_true and y_pred)
            y_true_flat = y.cpu().numpy().reshape(-1)  # 展平为一维数组
            y_pred_flat = pred.cpu().numpy().reshape(-1)  # 展平为一维数组
            r2 = r2_score(y_true_flat, y_pred_flat)
            eval_r2.append(r2)

            # 保存结果
            res.append(torch.cat((y, pred.cpu()), 1).detach().numpy())

    print("Finished Testing")
    # 输出各种指标
    test_rmse = np.sqrt(np.mean(eval_mse))
    test_mae = np.mean(eval_mae)
    test_mape = np.mean(eval_mape)
    test_smape = np.mean(eval_smape)
    test_r2 = np.mean(eval_r2)

    print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}, SMAPE: {test_smape:.4f}, R²: {test_r2:.4f}")
    
    return test_rmse, test_mae, test_mape, test_smape, test_r2
