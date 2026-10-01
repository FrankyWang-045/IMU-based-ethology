import numpy as np
import matplotlib.pyplot as plt

hmm = np.load("output/A5_C5_C5-c8_hmm.npz")
lat = np.load("output/A5_C5_C5-c8_latent.npz")

states = hmm["states"]
time = hmm["time"]
z = lat["downstream"]          # (T, 6)

# 截取一段（比如 20 秒）细看
t0, t1 = 500, 520
mask = (time >= t0) & (time < t1)

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

axes[0].plot(time[mask], z[mask])
axes[0].set_ylabel("latent z")

# 状态色带：用 scatter 或 pcolormesh 把状态画成色条
axes[1].scatter(time[mask], states[mask], c=states[mask],
                cmap="tab10", s=2, vmin=-0.5, vmax=9.5)
axes[1].set_ylabel("state")
axes[1].set_yticks(range(6))

plt.tight_layout()
plt.savefig("output/hmm_states_zoom.png", dpi=150)
plt.show()