import math

import random

from collections import OrderedDict

from torch.utils.data import DataLoader
import datetime
import os
import pickle

from datetime import datetime as dt
from multiprocessing import Pool
from torch import optim, nn

import numpy as np
import pandas as pd
import torch

class pxCountDatasetV2(torch.utils.data.Dataset):
    def __init__(self, path="./data/all", seq=[], grid_size=(18, 22), time_interval='30min', force_rebuild=False):
        assert len(seq) > 0
        if force_rebuild and os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.mkdir(path)
        self.seq = seq
        self.maxSeq = np.max(seq)
        self.path = path
        self.grid_size = grid_size
        self.time_interval = time_interval
        self.steps_per_day = {'15min': 96, '30min': 48, '1h': 24}[self.time_interval]

        self.lon_min, self.lon_max = 108.84, 109.05
        self.lat_min, self.lat_max = 34.19, 34.36
        self.d_lon = np.linspace(self.lon_min, self.lon_max, self.grid_size[1] + 1)
        self.d_lat = np.linspace(self.lat_min, self.lat_max, self.grid_size[0] + 1)

        self._init_pxCount()

    def lonlat_to_grid(self, lon, lat):
        lon_idx = np.digitize(lon, self.d_lon) - 1
        lat_idx = np.digitize(lat, self.d_lat) - 1
        lon_idx = np.clip(lon_idx, 0, self.grid_size[1] - 1)
        lat_idx = np.clip(lat_idx, 0, self.grid_size[0] - 1)

        # 修正：图像索引从上往下，纬度从下往上，需要翻转行号
        lat_idx = self.grid_size[0] - 1 - lat_idx
        return lon_idx, lat_idx

    def _init_dataset(self, date_in):
        tmp_date = date_in
        print("begin to preprocess the raw data of " + tmp_date)
        print("load POI and ext data")

        dataPOI = pd.read_csv("./data/xianPOI_agg.txt", sep="\t") \
            .query("(lat2 <= 34.36) and (lat2 >= 34.19) and (lon2>= 108.84) and (lon2 <= 109.05)")
        res = []
        for i in dataPOI.columns[2:]:
            tmp = dataPOI.loc[:, ["lon2", "lat2", i]]
            tmp0 = np.zeros(self.grid_size)
            for lon2, lat2, val in zip(tmp["lon2"], tmp["lat2"], tmp[i]):
                lon_idx, lat_idx = self.lonlat_to_grid(lon2, lat2)
                tmp0[lat_idx][lon_idx] = val
            res.append(tmp0)
        imgPOI = np.array(res)

        ext_data = pd.read_excel("./data/天气数据.xlsx", "Cleaned")
        tmp_wea = {v: k for k, v in enumerate(['晴', '多云', '阴', '小雨', '中雨', '大雨', '暴雨'])}
        tmp_wea = [max(tmp_wea.get(_[0], 0), tmp_wea.get(_[1], 0)) for x, _ in ext_data[["天气1", "天气2"]].iterrows()]
        ext_data["weather"] = [dict(enumerate(['晴', '多云', '阴', '小雨', '中雨', '大雨', '暴雨'])).get(_) for _ in tmp_wea]
        ext_data = ext_data[["date", "day_of_week", "maximum_temperature", "minimum_temperature",
                             "weather", "wind_direction", "wind_level", "AQI", "AQI_level"]]
        ext_data = pd.get_dummies(ext_data,
                                  columns=["day_of_week", "weather", "wind_direction", "wind_level", "AQI_level"])
        imgExt = {}
        for i in [dt.strftime(x, '%Y-%m-%d') for x in list(pd.date_range("2021-06-25", "2021-10-29"))]:
            tmp = ext_data[ext_data["date"] == i].drop(columns="date").to_numpy().squeeze()
            imgExt[i] = np.expand_dims(tmp, [-2, -1])

        list_dir = []
        tmp = [dt.strftime(x, '%Y-%m-%d') for x in list(pd.date_range("2021-06-26", "2021-10-29"))]
        for root, sub, files in os.walk("./data/pxnCount"):
            if "part-00000" in files and any([_ in root for _ in tmp]):
                if any([_ in root for _ in ["08-10", "08-11", "08-12", "08-13", "08-14", "08-15", "08-16", "08-17", "08-18"]]) and ("allOper" not in root):
                    continue
                list_dir.append(root + "/part-00000")
        list_dir = [s for s in list_dir if tmp_date in s]

        df = pd.DataFrame()
        for file in list_dir:
            df = pd.concat([df, pd.read_csv(file, sep="\t", names=["lon", "lat", "rDate", "rDateTime", "count", "uCount"])])
        pxCount = df.groupby(["lon", "lat", "rDate", "rDateTime"], as_index=False).sum() \
            .query("(lat <= 34.36) and (lat >= 34.19) and (lon >= 108.84) and (lon <= 109.05)").copy()

        for rDateTime in range(self.steps_per_day):
            tmp = pxCount.loc[(pxCount["rDate"] == tmp_date) & (pxCount["rDateTime"] == rDateTime), :]
            tmp0 = np.zeros(self.grid_size)
            for lon, lat, val in zip(tmp["lon"], tmp["lat"], tmp["uCount"]):
                lon_idx, lat_idx = self.lonlat_to_grid(lon, lat)
                tmp0[lat_idx][lon_idx] = val
            res = dict(img=tmp0, POI=imgPOI, ext=imgExt.get(tmp_date),
                       lastExt=imgExt.get((dt.strptime(tmp_date, '%Y-%m-%d') -
                                           datetime.timedelta(days=1)).strftime('%Y-%m-%d'), "no_data"))
            tmp_path = self.path + "/" + tmp_date + "-" + "{:0>2d}.pkl".format(rDateTime)
            with open(tmp_path, "wb") as pkl:
                pickle.dump(res, pkl)

    def _init_pxCount(self):
        myPool = Pool(os.cpu_count() - 1)
        if len(os.listdir(self.path)) == 0:
            tmp_list = [dt.strftime(x, '%Y-%m-%d') for x in list(pd.date_range("2021-06-26", "2021-10-29"))]
            myPool.map(self._init_dataset, tmp_list)
            myPool.close()
            myPool.join()
        elif len(os.listdir(self.path)) == self.steps_per_day * len(pd.date_range("2021-06-26", "2021-10-29")):
            print("raw data (in ./data/all) exist and have been preprocessed!")
        else:
            raise RuntimeError("the ./data/all is not empty")

    def __getitem__(self, index):
        list_dir = [os.path.join(self.path, _) for _ in os.listdir(self.path)]
        list_dir.sort()
        list_dir2 = list_dir[self.maxSeq:]

        if isinstance(index, slice):
            result = []
            for idx in range(*index.indices(len(list_dir2))):
                result.append(self._load_single_item(list_dir2[idx], list_dir))
            return result
        else:
            return self._load_single_item(list_dir2[index], list_dir)

    def _load_single_item(self, file, list_dir):
        res_dict = {}
        with open(file, 'rb') as f:
            tmp_f = pickle.load(f)
        res_dict["y"] = tmp_f["img"][None, :, :]
        res_dict["x_POI"] = tmp_f["POI"]
        res_dict["x_ext"] = np.array(tmp_f["ext"], dtype=np.float32)
        res_dict["x_lastExt"] = np.array(tmp_f["lastExt"], dtype=np.float32)

        seq_imgs = []
        for s in self.seq:
            file1 = list_dir[list_dir.index(file) - s]
            with open(file1, 'rb') as f:
                seq_imgs.append(pickle.load(f)["img"])
        res_dict["x_img"] = np.array(seq_imgs)
        return res_dict

    def __len__(self):
        return len(os.listdir(self.path)) - self.maxSeq
