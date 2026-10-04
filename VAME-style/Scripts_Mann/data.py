from pathlib import Path
import pandas as pd


class Data:
    def __init__(self, file_path):
        self.path = Path(file_path)
        self.name = self.path.stem
        self.df = self._load_csv(self.path)

        self.record_name = self.name.split("_")[0]+ "_" + self.name.split("_")[1]
        self.individual_name = self.name.split("_")[2]

        # 分模态读取数据
        self.time = self.df["time"].to_numpy()
        self.acc = self.df[["ax", "ay", "az"]].to_numpy()
        self.gyro = self.df[["wx", "wy", "wz"]].to_numpy()
        self.interp_mask = self.df["interp_mask"].to_numpy()


    def _load_csv(self, path):  #pandas读取csv文件
        df = pd.read_csv(path)
        return df


#批量加载文件夹中的 IMU 文件，返回 IMURecording 列表。
def load_recordings(folder, pattern="*.csv"):
    folder = Path(folder)
    files = folder.glob(pattern)  #识别该路径下所有.csv格式文件

    recordings = [Data(f) for f in files]

    return recordings




if __name__ == "__main__":
    recordings = load_recordings("Test_File/")
    for rec in recordings:
        print(rec.record_name, rec.individual_name, rec.time.shape)

