"""
IMU-Video Time Alignment Tool (lightweight Matplotlib GUI)
==========================================================

Align IMU and video timelines interactively, using the mouse-drop event
(mouse falling into the arena) as the synchronization signal.
Outputs the time offset t_diff.

Sign convention (consistent with imu_processor.data_preprocess)
---------------------------------------------------------------
    t_diff = IMU event time - video event time
i.e. IMU time minus t_diff aligns with video time
(in data_preprocess: ta = t_new_0 - t_diff)

Usage
-----
1. Recommended (notebook, shared raw variable)::

       %matplotlib tk
       from imu_processor import data_import
       from aligner import IMUVideoAligner

       raw = data_import(r"path/to/raw_csv_dir")   # load once
       aligner = IMUVideoAligner(
           r"path/to/video.mp4",
           raw_data=raw,       # reuse the pre-loaded raw array
       )
       aligner.show()
       t_diff = aligner.offset

2. Standalone / CLI (Aligner loads the IMU data itself)::

       python Aligner.py <video_path> <imu_raw_dir>

3. Direct instantiation with a directory::

       aligner = IMUVideoAligner(video_path, imu_dir=r"path/to/raw_csv_dir")

Controls
--------
- Video: slider / step buttons / Left-Right arrow keys (frame step),
  Space = play/pause
- IMU panel shows |a| in a zoomed window (default 30 s):
    Left-drag            : pan the view
    Right-drag / Shift+drag : shift IMU curve (adjust alignment)
    Scroll wheel         : zoom in/out around cursor
    Shift+Left/Right     : nudge t_diff by one video frame
- Workflow: navigate to the landing frame, click "1. Mark video event";
  then click "2. Mark IMU event" and click the landing spike on the curve.
  "Save t_diff" writes the result to t_diff.txt next to the video.

Notes
-----
- Only the first IMU_DURATION seconds of IMU data are used for display
  (speed). When raw_data is passed in, the full array is kept untouched;
  only the internal display copy is truncated.
- The first VIDEO_DURATION seconds of the video are extracted ONCE into a
  folder of 360p JPEG frames (cached next to the video). The GUI displays
  JPEGs only - no video codec is involved at runtime, which avoids black
  screens caused by OpenCV/MSMF codec issues on Windows.
- Requires: numpy, matplotlib, opencv-python,
  and imu_processor.py in the same directory.
"""

import os
import sys
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imu_processor import data_import, data_preprocess

# =============================================================================
# CONFIG (overridable via constructor arguments or command-line arguments)
# =============================================================================
DISPLAY_LPF_FC = 20.0               # 显示曲线低通截止频率 (Hz)
IMU_DURATION = 120.0                # only use the first N seconds of IMU data
VIEW_WINDOW = 30.0                  # initial width of the IMU view window (s)
WIN_SIZE = (11.0, 6.2)              # figure size in inches (fits 1080p screens)
VIDEO_DURATION = 120.0              # only the first N seconds of video are used
                                    # (None = full length)
PROXY_HEIGHT = 360                  # cached frame height in px (one-time extraction
                                    # of JPEG frames, cached next to the video)


def _prepare_frame_cache(video_path, max_duration=VIDEO_DURATION,
                         height=PROXY_HEIGHT):
    """
    Decode the first `max_duration` seconds of the video ONCE into a folder
    of downscaled JPEG frames. Displaying JPEGs needs no video codec at all,
    so seeking/playback in the GUI is fast and immune to codec/backend issues
    (e.g. OpenCV VideoWriter/MSMF failures on Windows).

    Frames correspond 1:1 to the original video at the same fps, so the time
    axis (and thus t_diff) is identical.

    Returns (frame_dir, fps, n_frames).
    """
    stem, _ = os.path.splitext(video_path)
    frame_dir = f"{stem}_{height}p_frames"
    meta_path = os.path.join(frame_dir, "meta.json")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = total if max_duration is None else min(total, int(round(max_duration * fps)))

    # reuse an existing complete cache
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            last = os.path.join(frame_dir, f"f{n - 1:06d}.jpg")
            if meta.get("n_frames") == n and meta.get("height") == height \
                    and os.path.isfile(last):
                print(f"Using cached frames: {frame_dir}")
                cap.release()
                return frame_dir, fps, n
        except Exception:
            pass

    os.makedirs(frame_dir, exist_ok=True)
    print(f"Extracting first {n / fps:.0f} s of video as {height}p JPEG frames "
          f"({n} frames, one-time only)...")
    idx = 0
    while idx < n:
        ok, frame = cap.read()
        if not ok:
            print(f"WARNING: decode stopped early at frame {idx}")
            break
        h, w = frame.shape[:2]
        if h > height:
            new_w = int(round(w * height / h / 2) * 2)   # keep width even
            frame = cv2.resize(frame, (new_w, height))
        cv2.imwrite(os.path.join(frame_dir, f"f{idx:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        idx += 1
        if idx % 300 == 0:
            print(f"  {idx}/{n} frames ({100.0 * idx / n:.0f}%)")
    cap.release()
    n = idx
    with open(meta_path, "w") as f:
        json.dump({"source": os.path.basename(video_path), "height": height,
                   "fps": fps, "n_frames": n}, f)
    print(f"Frame cache saved: {frame_dir} ({n} frames)")
    return frame_dir, fps, n


class IMUVideoAligner:
    """Interactive IMU-video time alignment GUI.

    Parameters
    ----------
    video_path : str
        Video file path.
    raw_data : np.ndarray, shape (n, >=8), optional
        Pre-loaded raw IMU data (output of ``imu_processor.data_import``).
        Recommended: share the same array with the final preprocessing step
        so the data is read from disk only once.
    imu_dir : str, optional
        Directory of raw IMU CSV files. Only used when ``raw_data`` is None.
    lpf_fc : float
        显示曲线低通截止频率 (Hz)。与最终预处理一致（默认 20Hz）。
    imu_duration : float
        Only the first N seconds of IMU data are shown (speed).
    view_window : float
        Initial width of the IMU view window (s).
    """

    def __init__(self, video_path, raw_data=None, imu_dir=None,
                 lpf_fc=DISPLAY_LPF_FC, imu_duration=IMU_DURATION,
                 view_window=VIEW_WINDOW):
        # ---------------- video ----------------
        # keep the original path for t_diff.txt output; display JPEG frame cache
        self.video_path = video_path
        self.frame_dir, self.fps, self.n_frames = _prepare_frame_cache(video_path)
        self.frame_idx = 0
        self.playing = False
        self._pending_frame = None   # debounced slider target

        # ---------------- IMU (first imu_duration seconds only) ----------------
        # 优先使用外部传入的 raw 数组（与最终预处理共用同一份数据）；
        # 否则从 imu_dir 自行读取。
        if raw_data is not None:
            raw = np.asarray(raw_data, dtype=float)
            print("Using pre-loaded raw IMU data "
                  f"(shape {raw.shape}, first {imu_duration:.0f} s shown).")
        elif imu_dir is not None:
            print(f"Loading IMU data from {imu_dir} "
                  f"(first {imu_duration:.0f} s only)...")
            raw = data_import(imu_dir, verbose=True)
        else:
            raise ValueError(
                "Either raw_data (pre-loaded array) or imu_dir (CSV directory) "
                "must be provided.")

        t0 = raw[0, 0]
        raw_disp = raw[raw[:, 0] - t0 <= imu_duration + 5.0]   # small margin
        # t_diff=0.0: keep the original IMU time base; offset is set by the GUI
        proc = data_preprocess(raw_disp, t_diff=0.0, apply_lpf=True, lpf_fc=lpf_fc)
        proc = proc[proc[:, 0] <= imu_duration]
        self.t_imu = proc[:, 0]                     # original IMU time axis
        self.acc_norm = np.linalg.norm(proc[:, 1:4], axis=1)

        # ---------------- alignment state ----------------
        self.offset = 0.0            # current t_diff
        self.t_video_event = None
        self.t_imu_event = None
        self._mode = None            # None | 'pan' | 'shift'
        self._mark_imu_armed = False

        # ---------------- view state ----------------
        self.view_x0 = 0.0
        self.view_w = view_window

        self._build_gui()
        self._refresh()

    # ------------------------------------------------------------------ GUI
    def _build_gui(self):
        self.fig = plt.figure(figsize=WIN_SIZE, dpi=100)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("IMU-Video Time Alignment")

        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1],
                                   left=0.06, right=0.97, top=0.90,
                                   bottom=0.30, hspace=0.30)
        # video panel
        self.ax_video = self.fig.add_subplot(gs[0])
        self.ax_video.set_xticks([]); self.ax_video.set_yticks([])
        self.im_video = self.ax_video.imshow(self._read_frame(0))

        # IMU panel
        self.ax_imu = self.fig.add_subplot(gs[1])
        (self.line_imu,) = self.ax_imu.plot(self.t_imu, self.acc_norm,
                                            lw=0.8, color="C0")
        self.ax_imu.set_xlabel("time (s)   [IMU time - t_diff]")
        self.ax_imu.set_ylabel("|a| (g)")
        self.ax_imu.set_title(
            "IMU |a|  (left-drag: pan, right-drag/shift-drag: shift alignment, "
            "scroll: zoom; red line = current video time)")
        self.ax_imu.grid(alpha=0.3)
        self.vline = self.ax_imu.axvline(0.0, color="r", lw=1.2)
        (self.mark_v,) = self.ax_imu.plot([], [], "rv", ms=10, label="video event")
        (self.mark_i,) = self.ax_imu.plot([], [], "g^", ms=10, label="IMU event")
        self.ax_imu.legend(loc="upper right")

        # status + help text
        self.status = self.fig.text(0.06, 0.955, "", fontsize=10)
        self.fig.text(
            0.06, 0.925,
            "Drag: pan view | Right-drag / Shift+drag: shift alignment | "
            "Scroll: zoom | Left/Right: frame step | "
            "Shift+Left/Right: nudge t_diff 1 frame | Space: play/pause",
            fontsize=8, color="gray")

        # video frame slider
        ax_slider = self.fig.add_axes([0.06, 0.22, 0.88, 0.03])
        self.slider = Slider(ax_slider, "frame", 0, self.n_frames - 1,
                             valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        # frame step buttons
        labels = ["-100", "-10", "-1", "Play/Pause", "+1", "+10", "+100"]
        steps = [-100, -10, -1, None, 1, 10, 100]
        self._step_btns = []
        for i, (lab, st) in enumerate(zip(labels, steps)):
            ax_b = self.fig.add_axes([0.06 + i * 0.075, 0.15, 0.07, 0.045])
            b = Button(ax_b, lab)
            if st is None:
                b.on_clicked(self._toggle_play)
            else:
                b.on_clicked(lambda e, s=st: self._step_frame(s))
            self._step_btns.append(b)

        # operation buttons
        ops = [("1. Mark video event", self._mark_video),
               ("2. Mark IMU event", self._arm_mark_imu),
               ("Reset", self._reset),
               ("Save t_diff", self._save)]
        self._op_btns = []
        for i, (lab, fn) in enumerate(ops):
            ax_b = self.fig.add_axes([0.60 + i * 0.095, 0.15, 0.09, 0.045])
            b = Button(ax_b, lab)
            b.on_clicked(fn)
            self._op_btns.append(b)

        # mouse / keyboard events
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # playback timer
        self.timer = self.fig.canvas.new_timer(interval=int(1000 / self.fps))
        self.timer.add_callback(self._on_timer)

        # slider debounce timer: dragging the slider fires many events,
        # only seek 150 ms after the last one (avoids 4K seek storms)
        self._seek_timer = self.fig.canvas.new_timer(interval=150)
        self._seek_timer.single_shot = True
        self._seek_timer.add_callback(self._do_pending_seek)

    # --------------------------------------------------------- video control
    def _read_frame(self, idx):
        # read a cached JPEG frame: no video codec involved, always reliable
        path = os.path.join(self.frame_dir, f"f{idx:06d}.jpg")
        frame = cv2.imread(path)
        if frame is None:
            frame = np.zeros((PROXY_HEIGHT, 640, 3), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _set_frame(self, idx):
        self.frame_idx = int(np.clip(idx, 0, self.n_frames - 1))
        self.im_video.set_data(self._read_frame(self.frame_idx))
        self.ax_video.set_title(
            f"frame {self.frame_idx}/{self.n_frames - 1}   "
            f"t = {self.frame_idx / self.fps:.3f} s")
        self.slider.eventson = False
        self.slider.set_val(self.frame_idx)
        self.slider.eventson = True
        self._refresh()

    def _step_frame(self, step):
        self._set_frame(self.frame_idx + step)

    def _on_slider(self, val):
        self._pending_frame = int(val)
        self._seek_timer.start()

    def _do_pending_seek(self):
        if self._pending_frame is not None:
            self._set_frame(self._pending_frame)
            self._pending_frame = None

    def _toggle_play(self, event=None):
        self.playing = not self.playing
        self.timer.start() if self.playing else self.timer.stop()

    def _on_timer(self):
        if self.frame_idx >= self.n_frames - 1:
            self._toggle_play()
            return
        self._set_frame(self.frame_idx + 1)

    def _on_key(self, event):
        if event.key == "left":
            self._step_frame(-1)
        elif event.key == "right":
            self._step_frame(1)
        elif event.key == "shift+left":
            self.offset -= 1.0 / self.fps
            self._refresh()
        elif event.key == "shift+right":
            self.offset += 1.0 / self.fps
            self._refresh()
        elif event.key == " ":
            self._toggle_play()

    # ------------------------------------------------------ alignment & view
    def _px_to_sec(self):
        """seconds per pixel on the IMU axes (for robust drag handling)"""
        return self.view_w / max(self.ax_imu.bbox.width, 1)

    def _on_press(self, event):
        if event.inaxes != self.ax_imu or event.xdata is None:
            return
        if self._mark_imu_armed:
            # mark mode: clicked point = IMU event
            # displayed x = original time - offset => original = x + offset
            self.t_imu_event = event.xdata + self.offset
            self._mark_imu_armed = False
            if self.t_video_event is not None:
                self.offset = self.t_imu_event - self.t_video_event
            self._refresh()
            return
        shift_held = bool(event.key) and "shift" in str(event.key)
        if event.button == 3 or (event.button == 1 and shift_held):
            self._mode = "shift"          # shift alignment
            self._drag_offset0 = self.offset
        elif event.button == 1:
            self._mode = "pan"            # pan view
            self._drag_view0 = self.view_x0
        self._drag_px0 = event.x

    def _on_motion(self, event):
        if self._mode is None or event.x is None:
            return
        dx = (event.x - self._drag_px0) * self._px_to_sec()
        if self._mode == "shift":
            # dragging the curve right (dx>0) => IMU time minus more => offset down
            self.offset = self._drag_offset0 - dx
            self._refresh()
        elif self._mode == "pan":
            self.view_x0 = self._drag_view0 - dx
            self._apply_view()
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        self._mode = None

    def _on_scroll(self, event):
        if event.inaxes != self.ax_imu or event.xdata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        new_w = float(np.clip(self.view_w * factor, 1.0, 600.0))
        frac = (event.xdata - self.view_x0) / self.view_w   # keep cursor anchored
        self.view_x0 = event.xdata - frac * new_w
        self.view_w = new_w
        self._apply_view()
        self._refresh()

    def _apply_view(self):
        self.ax_imu.set_xlim(self.view_x0, self.view_x0 + self.view_w)

    def _mark_video(self, event=None):
        self.t_video_event = self.frame_idx / self.fps
        if self.t_imu_event is not None:
            self.offset = self.t_imu_event - self.t_video_event
        self._refresh()

    def _arm_mark_imu(self, event=None):
        self._mark_imu_armed = True
        self._refresh()

    def _reset(self, event=None):
        self.offset = 0.0
        self.t_video_event = None
        self.t_imu_event = None
        self._mark_imu_armed = False
        self._refresh()

    def _save(self, event=None):
        out_path = os.path.join(os.path.dirname(os.path.abspath(self.video_path)),
                                "t_diff.txt")
        with open(out_path, "w") as f:
            f.write(f"{self.offset:.6f}\n")
        print(f"t_diff = {self.offset:.6f} s saved -> {out_path}")
        print(f"usage: data_preprocess(raw, t_diff={self.offset:.6f})")

    # --------------------------------------------------------------- refresh
    def _refresh(self):
        t_disp = self.t_imu - self.offset          # displayed time axis
        self.line_imu.set_xdata(t_disp)
        t_now = self.frame_idx / self.fps
        self.vline.set_xdata([t_now, t_now])

        y_top = np.nanmax(self.acc_norm)
        if self.t_video_event is not None:
            self.mark_v.set_data([self.t_video_event], [y_top])
        else:
            self.mark_v.set_data([], [])
        if self.t_imu_event is not None:
            self.mark_i.set_data([self.t_imu_event - self.offset], [y_top])
        else:
            self.mark_i.set_data([], [])

        # auto-follow: re-center view if the video cursor left the window
        if not (self.view_x0 <= t_now <= self.view_x0 + self.view_w):
            self.view_x0 = t_now - 0.2 * self.view_w
        self._apply_view()

        # autoscale y within the visible window only
        in_view = (t_disp >= self.view_x0) & (t_disp <= self.view_x0 + self.view_w)
        if in_view.any():
            y = self.acc_norm[in_view]
            pad = max(0.1 * (y.max() - y.min()), 0.05)
            self.ax_imu.set_ylim(y.min() - pad, y.max() + pad)

        # status line
        fmt = lambda v: "  --   " if v is None else f"{v:8.3f}s"
        armed = "   [click the IMU curve...]" if self._mark_imu_armed else ""
        self.status.set_text(
            f"video event: {fmt(self.t_video_event)}   "
            f"IMU event: {fmt(self.t_imu_event)}   "
            f"t_diff = {self.offset:+.3f} s "
            f"(usage: data_preprocess(raw, t_diff={self.offset:.3f})){armed}")
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# =============================================================================
# entry point
# =============================================================================

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    imu_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if not video_path:
        raise SystemExit(
            "Usage: python aligner.py <video_path> <imu_raw_dir>")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if imu_dir is not None and not os.path.isdir(imu_dir):
        raise FileNotFoundError(f"IMU directory not found: {imu_dir}")

    aligner = IMUVideoAligner(video_path, imu_dir=imu_dir)
    aligner.show()
