# import numpy as np, glob
# files = glob.glob(r"F:\IMU_computational_ethology\Data\VAME-Style-Data\smalltest\*_latent.npz")
# codes = np.concatenate([np.load(f)["codes"] for f in files])
# print("码本利用率:", len(np.unique(codes)), "/ 32")
# dwell = np.diff(np.where(np.diff(codes) != 0)[0]).mean()
# print(f"码平均驻留: {dwell:.1f} 帧")


import pickle, numpy as np
# m = pickle.load(open(r"F:\IMU_computational_ethology\Data\VAME-Style-Data\smalltest\hmm_cat_model.pkl", "rb"))
# E = m.emissionprob_
# sim = E @ E.T / (np.linalg.norm(E, axis=1, keepdims=True) * np.linalg.norm(E, axis=1))
# for i in range(28):
#     for j in range(i+1, 28):
#         if sim[i, j] > 0.9:
#             print(f"状态 {i} ~ {j}: 相似度 {sim[i,j]:.3f}")


OUT = r"F:\IMU_computational_ethology\Data\VAME-Style-Data\smalltest"
for name in ["A1_C1_A1-6d", "A1_C1_C1-1c"]:
    d = np.load(f"{OUT}/{name}_latent.npz")
    codes, z = d["codes"], d["downstream"]
    wz_proxy = np.abs(z).mean(axis=1)          # 静止时 latent 各维都趋稳
    quiet = np.sort(np.argpartition(wz_proxy, 5000)[:5000])
    qc = codes[quiet]
    print(f"{name}: 静止段唯一码数 {len(np.unique(qc))}, "
          f"切换率 {(np.diff(qc) != 0).mean():.1%}")