m = pickle.load(open(r"F:\IMU_computational_ethology\Data\VAME-Style-Data\smalltest\hmm_cat_model.pkl", "rb"))
# E = m.emissionprob_
# sim = E @ E.T / (np.linalg.norm(E, axis=1, keepdims=True) * np.linalg.norm(E, axis=1))
# for i in range(28):
#     for j in range(i+1, 28):
#         if sim[i, j] > 0.9:
#             print(f"状态 {i} ~ {j}: 相似度 {sim[i,j]:.3f}")

