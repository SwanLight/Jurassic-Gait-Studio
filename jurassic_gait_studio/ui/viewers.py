from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from ..core.io import load_skeleton_csv
from ..core.morphology_preview import build_species_pose
from ..core.render import BG_COLOR, default_camera_for_sequence, render_scene_3d
from ..core.schema import SkeletonSequence, SpeciesMorphology

BG = "#0B0F14"
PANEL = "#151F2A"
BORDER = "#304150"
TEXT = "#F6E1B3"
MUTED = "#A7B7C7"
ACCENT = "#D99023"
ACCENT_HOVER = "#F0B34C"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _crop_background(arr: np.ndarray, bg_hex: str = BG_COLOR, tol: int = 6) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    bg = np.array(_hex_to_rgb(bg_hex), dtype=np.int16)
    rgb = arr[..., :3].astype(np.int16)
    diff = np.abs(rgb - bg[None, None, :]).max(axis=2)
    mask = diff > tol
    if not np.any(mask):
        return arr
    ys, xs = np.where(mask)
    pad = 10
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
    return arr[y0:y1, x0:x1]


def _viewer_camera(sequence: SkeletonSequence, kind: str) -> dict[str, float | str | bool]:
    return dict(default_camera_for_sequence(sequence))


class Interactive3DViewer(ctk.CTkToplevel):
    def __init__(self, parent: ctk.CTkBaseClass, sequence: SkeletonSequence, title: str = "3D Viewer", kind: str = "gait") -> None:
        super().__init__(parent)
        self.sequence = sequence
        self.kind = kind
        self.title(title)
        self.geometry("1240x860")
        self.minsize(980, 700)
        self.configure(fg_color=BG)
        self.frame_var = ctk.IntVar(value=0)
        self.playing = kind == "gait" and sequence.xyz.shape[0] > 1
        self.interval_ms = 70
        self._bounds = None
        self._camera = _viewer_camera(sequence, kind)
        self._has_mapped_once = False
        self._build()
        from ..core.render import _sequence_bounds

        self._bounds = _sequence_bounds(sequence)
        self._setup_scene()
        self.bind("<Map>", self._on_first_map, add="+")
        self.update_idletasks()
        self.after(80, self.force_refresh)
        self.after(self.interval_ms, self._tick)

    @classmethod
    def from_csv(cls, parent: ctk.CTk, csv_path: str | Path, title: str | None = None) -> "Interactive3DViewer":
        return cls(parent, sequence=load_skeleton_csv(csv_path), title=title or f"3D Viewer • {Path(csv_path).stem}", kind="gait")

    @classmethod
    def from_species(cls, parent: ctk.CTk, species: SpeciesMorphology, title: str | None = None) -> "Interactive3DViewer":
        return cls(parent, sequence=build_species_pose(species), title=title or f"3D Morphology • {species.name}", kind="morphology")

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=18, border_width=1, border_color=BORDER)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 10))
        header.grid_columnconfigure(1, weight=1)

        heading = "Interactive 3D gait viewer" if self.kind == "gait" else "Static 3D morphology preview"
        subtitle = (
            "Drag to orbit, scroll to zoom, and let the dinosaur keep walking in loop."
            if self.kind == "gait"
            else "Drag to orbit, scroll to zoom, and inspect the current standard pose while editing parameters."
        )
        self.header_title = ctk.CTkLabel(header, text=heading, font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT)
        self.header_title.grid(row=0, column=0, sticky="w", padx=16, pady=(14, 4))
        self.header_subtitle = ctk.CTkLabel(header, text=subtitle, text_color=MUTED)
        self.header_subtitle.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 14))
        btns = ctk.CTkFrame(header, fg_color="transparent")
        btns.grid(row=0, column=1, rowspan=2, sticky="e", padx=16)
        self.play_btn = ctk.CTkButton(btns, text="Pause" if self.playing else "Play", fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#120B02", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=(0, 8))
        if self.kind != "gait":
            self.play_btn.configure(state="disabled")
        ctk.CTkButton(btns, text="Save current frame", fg_color="#2B3440", hover_color="#3C4958", command=self.save_current_frame).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btns, text="Reset view", fg_color="#2B3440", hover_color="#3C4958", command=self.reset_view).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btns, text="Close", fg_color="#74332A", hover_color="#8E4338", command=self.destroy).pack(side="left")

        body = ctk.CTkFrame(self, fg_color=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(10.8, 6.8), dpi=100, facecolor=BG)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=body)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar_frame = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
        toolbar_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        controls = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=16, border_width=1, border_color=BORDER)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        controls.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(controls, text="Frame", text_color=TEXT).grid(row=0, column=0, sticky="w", padx=14, pady=12)
        slider_to = max(1, int(self.sequence.xyz.shape[0] - 1))
        slider_steps = max(1, int(self.sequence.xyz.shape[0] - 1))
        self.frame_slider = ctk.CTkSlider(
            controls,
            from_=0,
            to=slider_to,
            number_of_steps=slider_steps,
            variable=self.frame_var,
            command=self._slider_changed,
        )
        self.frame_slider.grid(row=0, column=1, sticky="ew", padx=(0, 14), pady=12)
        self.frame_label = ctk.CTkLabel(controls, text="0", text_color=MUTED)
        self.frame_label.grid(row=0, column=2, sticky="e", padx=14)
        self._refresh_controls_state()

    def _on_first_map(self, _event=None) -> None:
        if self._has_mapped_once:
            return
        self._has_mapped_once = True
        self.after(30, self.force_refresh)

    def force_refresh(self) -> None:
        try:
            self._draw_frame(int(self.frame_var.get()))
            self.canvas.draw_idle()
            self.update_idletasks()
        except Exception:
            pass

    def _refresh_controls_state(self) -> None:
        is_static = self.sequence.xyz.shape[0] <= 1 or self.kind != "gait"
        if is_static:
            self.frame_slider.configure(state="disabled")
            self.frame_label.configure(text="Static pose")
        else:
            self.frame_slider.configure(state="normal")
            self.frame_label.configure(text=str(int(self.frame_var.get())))

    def _setup_scene(self) -> None:
        self.ax.set_facecolor(BG)
        self.reset_view(redraw=False)
        self.ax.grid(False)

    def reset_view(self, redraw: bool = True) -> None:
        self._camera = _viewer_camera(self.sequence, self.kind)
        self.ax.view_init(elev=float(self._camera["elev"]), azim=float(self._camera["azim"]))
        if hasattr(self.ax, "dist"):
            try:
                self.ax.dist = float(self._camera["distance"])
            except Exception:
                pass
        if redraw:
            self.canvas.draw_idle()

    def set_sequence(self, sequence: SkeletonSequence, *, title: str | None = None, kind: str | None = None) -> None:
        from ..core.render import _sequence_bounds

        self.sequence = sequence
        if title:
            self.title(title)
        if kind:
            self.kind = kind
        self.playing = self.kind == "gait" and self.sequence.xyz.shape[0] > 1
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        self.play_btn.configure(state="normal" if self.kind == "gait" else "disabled")
        self.frame_var.set(0)
        self.frame_slider.configure(to=max(1, int(self.sequence.xyz.shape[0] - 1)), number_of_steps=max(1, int(self.sequence.xyz.shape[0] - 1)))
        self._bounds = _sequence_bounds(sequence)
        self._refresh_controls_state()
        self.reset_view(redraw=False)
        self.after(20, self.force_refresh)

    def toggle_play(self) -> None:
        if self.kind != "gait":
            return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")

    def _slider_changed(self, value: float) -> None:
        self._draw_frame(int(round(float(value))))

    def _tick(self) -> None:
        if self.playing and self.sequence.xyz.shape[0] > 1:
            frame = (int(self.frame_var.get()) + 1) % self.sequence.xyz.shape[0]
            self.frame_var.set(frame)
            self._draw_frame(frame)
        self.after(self.interval_ms, self._tick)

    def _draw_frame(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, self.sequence.xyz.shape[0] - 1))
        if self.sequence.xyz.shape[0] > 1 and self.kind == "gait":
            self.frame_label.configure(text=str(frame))
        else:
            self.frame_label.configure(text="Static pose")
        current_elev = getattr(self.ax, "elev", float(self._camera["elev"]))
        current_azim = getattr(self.ax, "azim", float(self._camera["azim"]))
        current_dist = getattr(self.ax, "dist", float(self._camera["distance"]))
        try:
            render_scene_3d(
                self.ax,
                self.sequence,
                frame,
                bounds=self._bounds,
                elev=float(current_elev),
                azim=float(current_azim),
                distance=float(current_dist),
                wall_side=str(self._camera.get("wall_side", "near")),
                invert_y=bool(self._camera.get("invert_y", False)),
            )
            self.canvas.draw()
        except Exception as exc:
            self.ax.cla()
            self.ax.text2D(0.5, 0.55, "3D preview failed", transform=self.ax.transAxes, ha="center", va="center")
            self.ax.text2D(0.5, 0.46, str(exc), transform=self.ax.transAxes, ha="center", va="center", fontsize=9)
            self.canvas.draw()

    def save_current_frame(self) -> None:
        try:
            self.figure.canvas.draw()
            rgba = np.asarray(self.figure.canvas.buffer_rgba()).copy()
            cropped = _crop_background(rgba)
            out_path = filedialog.asksaveasfilename(
                parent=self,
                title="Save current 3D frame",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png")],
            )
            if not out_path:
                return
            Image.fromarray(cropped).save(out_path)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
