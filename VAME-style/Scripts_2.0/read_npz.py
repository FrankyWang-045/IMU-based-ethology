import numpy as np

npz_path = r"F:\IMU_computational_ethology\Data\VAME-Style-Data\4.0smalltest\A1_C1_C1-1c.npz"

data = np.load(npz_path)

print(f"文件: {npz_path}")
for key in data.keys():
    arr = data[key]
    print(f"\nkey = '{key}'")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")

    if arr.ndim == 1:
        print(f"  前 5 行: {arr[:5]}")
    elif arr.ndim == 2:
        print(f"  前 5 行:\n{arr[:5]}")
    else:
        print(f"  前 5 个元素: {arr.flat[:5]}")