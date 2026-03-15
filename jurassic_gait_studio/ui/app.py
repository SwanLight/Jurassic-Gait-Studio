from __future__ import annotations

import argparse
import json
import queue
import shutil
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Callable, Dict, Iterable, List

import customtkinter as ctk
from PIL import Image, ImageGrab, ImageOps

from ..core.morphology_preview import build_species_pose, render_species_preview, species_from_payload, summarize_species_payload
from ..paths import ASSETS_DIR, META_DIR, RUNS_DIR
from ..registry import (
    bootstrap_workspace,
    get_species_path,
    import_bird_clip,
    import_species_json,
    infer_bird_species_from_filename,
    list_bird_clips,
    list_runs,
    list_species,
    save_species_payload,
)
from ..studio import generate_run, preview_weights
from .viewers import Interactive3DViewer

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG = "#0B0F14"
SURFACE = "#111821"
PANEL = "#151F2A"
CARD = "#1B2733"
CARD_ALT = "#20303F"
CARD_HOVER = "#2A3E52"
BORDER = "#304150"
TEXT = "#F6E1B3"
MUTED = "#A7B7C7"
ACCENT = "#D99023"
ACCENT_HOVER = "#858FAC"
VIOLET = "#544E70"
VIOLET_HOVER = "#76637E"
SUCCESS = "#53BB8A"
INFO = "#3E5E78"

LAB_QUICK_FIELDS = [
    {"id": "hip_width_m", "group": "Lengths", "label": "Hip width", "hint": "Left hip to right hip spacing.", "min": 0.02, "max": 1.8},
    {"id": "rear_body_length_m", "group": "Lengths", "label": "Rear body length", "hint": "Back body segment, from hip peak toward the tail root.", "min": 0.02, "max": 1.8},
    {"id": "front_body_length_m", "group": "Lengths", "label": "Front body length", "hint": "Front body segment, from hip peak toward the neck root.", "min": 0.04, "max": 2.4},
    {"id": "tail_length_m", "group": "Lengths", "label": "Tail length", "hint": "Tail rod from tail root to tail tip.", "min": 0.04, "max": 3.0},
    {"id": "neck_length_m", "group": "Lengths", "label": "Neck length", "hint": "Neck rod from shoulder end to head base.", "min": 0.01, "max": 1.2},
    {"id": "head_length_m", "group": "Lengths", "label": "Head length", "hint": "Head rod from head base to snout tip.", "min": 0.01, "max": 1.2},
    {"id": "thigh_length_m", "group": "Lengths", "label": "Thigh length", "hint": "Hip to knee segment.", "min": 0.02, "max": 1.2},
    {"id": "shank_length_m", "group": "Lengths", "label": "Shank length", "hint": "Knee to ankle segment.", "min": 0.02, "max": 1.2},
    {"id": "metatarsus_length_m", "group": "Lengths", "label": "Metatarsus length", "hint": "Ankle to toe-base segment.", "min": 0.01, "max": 1.0},
    {"id": "foot_length_m", "group": "Lengths", "label": "Foot length", "hint": "Toe-base to toe-tip segment.", "min": 0.01, "max": 0.9},

    {"id": "rear_body_diameter_m", "group": "Diameters", "label": "Rear body diameter", "hint": "Thickness of the rear body segment at the hips.", "min": 0.01, "max": 1.2},
    {"id": "front_body_diameter_m", "group": "Diameters", "label": "Front body diameter", "hint": "Thickness of the front body segment.", "min": 0.01, "max": 1.0},
    {"id": "tail_base_diameter_m", "group": "Diameters", "label": "Tail base diameter", "hint": "Thickness where the tail starts.", "min": 0.006, "max": 0.6},
    {"id": "tail_tip_diameter_m", "group": "Diameters", "label": "Tail tip diameter", "hint": "Thickness near the end of the tail.", "min": 0.003, "max": 0.3},
    {"id": "neck_diameter_m", "group": "Diameters", "label": "Neck diameter", "hint": "Thickness of the neck rod.", "min": 0.006, "max": 0.5},
    {"id": "head_diameter_m", "group": "Diameters", "label": "Head diameter", "hint": "Overall thickness of the head rod.", "min": 0.006, "max": 0.5},
    {"id": "thigh_diameter_m", "group": "Diameters", "label": "Thigh diameter", "hint": "Thickness of the upper leg.", "min": 0.006, "max": 0.5},
    {"id": "shank_diameter_m", "group": "Diameters", "label": "Shank diameter", "hint": "Thickness of the lower leg.", "min": 0.006, "max": 0.4},
    {"id": "metatarsus_diameter_m", "group": "Diameters", "label": "Metatarsus diameter", "hint": "Thickness of the ankle-to-toe-base rod.", "min": 0.004, "max": 0.3},

    {"id": "rear_body_angle_deg", "group": "Angles", "label": "Rear body angle", "hint": "Side-view up/down angle of the rear body segment.", "min": -20.0, "max": 35.0},
    {"id": "front_body_angle_deg", "group": "Angles", "label": "Front body angle", "hint": "Side-view up/down angle of the front body segment.", "min": -20.0, "max": 30.0},
    {"id": "tail_angle_deg", "group": "Angles", "label": "Tail angle", "hint": "Side-view up/down angle of the tail.", "min": -20.0, "max": 25.0},
    {"id": "neck_angle_deg", "group": "Angles", "label": "Neck angle", "hint": "Side-view up/down angle of the neck rod.", "min": -10.0, "max": 80.0},
    {"id": "head_angle_deg", "group": "Angles", "label": "Head angle", "hint": "Side-view up/down angle of the head rod.", "min": -20.0, "max": 50.0},
]



class ImagePlayer:
    def __init__(self, label: ctk.CTkLabel, size: tuple[int, int], fill_bg: str = BG) -> None:
        self.label = label
        self.base_size = size
        self.fill_bg = fill_bg
        self.frames: List[ctk.CTkImage] = []
        self._raw_frames: List[Image.Image] = []
        self._after_id: str | None = None
        self._resize_after: str | None = None
        self._frame_idx = 0
        self.label.bind("<Configure>", self._schedule_resize, add="+")

    def clear(self, text: str = "") -> None:
        self.stop()
        self._raw_frames = []
        self.label.configure(image=None, text=text)

    def stop(self) -> None:
        if self._after_id:
            try:
                self.label.after_cancel(self._after_id)
            except Exception:
                pass
        self._after_id = None
        self.frames = []
        self._frame_idx = 0

    def _target_size(self) -> tuple[int, int]:
        try:
            width = max(160, self.label.winfo_width() - 12)
            height = max(120, self.label.winfo_height() - 12)
        except Exception:
            return self.base_size
        if width <= 170 and height <= 130:
            width, height = self.base_size
        return width, height

    def _fit(self, image: Image.Image) -> Image.Image:
        size = self._target_size()
        frame = image.convert("RGBA")
        frame.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", size, self.fill_bg)
        x = (size[0] - frame.size[0]) // 2
        y = (size[1] - frame.size[1]) // 2
        canvas.paste(frame, (x, y), frame)
        return canvas

    def load_gif(self, path: str | Path) -> None:
        self.stop()
        img = Image.open(path)
        self._raw_frames = []
        try:
            while True:
                self._raw_frames.append(img.copy().convert("RGBA"))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        self._rebuild_frames(animated=True)

    def load_image(self, path: str | Path) -> None:
        self.stop()
        self._raw_frames = [Image.open(path).convert("RGBA")]
        self._rebuild_frames(animated=False)

    def _rebuild_frames(self, animated: bool | None = None) -> None:
        if not self._raw_frames:
            self.clear("No preview available")
            return
        if animated is None:
            animated = len(self._raw_frames) > 1
        self.frames = []
        size = self._target_size()
        for raw in self._raw_frames:
            canvas = self._fit(raw)
            self.frames.append(ctk.CTkImage(light_image=canvas, dark_image=canvas, size=size))
        self._frame_idx = 0
        if animated:
            self._tick()
        else:
            self.label.configure(image=self.frames[0], text="")

    def _schedule_resize(self, _event=None) -> None:
        if self._resize_after:
            try:
                self.label.after_cancel(self._resize_after)
            except Exception:
                pass
        self._resize_after = self.label.after(120, self._on_resize)

    def _on_resize(self) -> None:
        self._resize_after = None
        if self._raw_frames:
            self.stop()
            self._rebuild_frames(animated=len(self._raw_frames) > 1)

    def _tick(self) -> None:
        if not self.frames:
            return
        self.label.configure(image=self.frames[self._frame_idx], text="")
        self._frame_idx = (self._frame_idx + 1) % len(self.frames)
        self._after_id = self.label.after(95, self._tick)


class FusionBarsPanel(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkBaseClass) -> None:
        super().__init__(master, fg_color="#10161D", corner_radius=18, border_width=1, border_color=BORDER)
        self.grid_columnconfigure(0, weight=1)
        self.report_rows: List[Dict[str, Any]] = []
        self.animation_progress = 0.0
        self.anim_after: str | None = None

        self.header = ctk.CTkLabel(self, text="Automatic bird weighting", font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT)
        self.header.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))
        self.subtitle = ctk.CTkLabel(self, text="Preview weights to inspect compatibility.", text_color=MUTED, justify="left")
        self.subtitle.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.canvas = tk.Canvas(self, height=240, bg="#10161D", highlightthickness=0, bd=0)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda _e: self._draw_chart())
        self.summary = ctk.CTkLabel(self, text="", text_color=MUTED, justify="left")
        self.summary.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))

    def clear(self, text: str = "Preview weights to inspect compatibility.") -> None:
        if self.anim_after:
            try:
                self.after_cancel(self.anim_after)
            except Exception:
                pass
        self.anim_after = None
        self.report_rows = []
        self.animation_progress = 0.0
        self.subtitle.configure(text=text)
        self.summary.configure(text="")
        self._draw_chart()

    def set_report(self, report: Any) -> None:
        self.report_rows = list(getattr(report, "contributions", []))
        target_name = getattr(report, "target_species", "target")
        n_ref = getattr(report, "n_reference_birds", len(self.report_rows))
        self.subtitle.configure(text=f"Target: {target_name} • {n_ref} reference bird(s)")
        if self.report_rows:
            leader = max(self.report_rows, key=lambda row: float(row.get("weight", 0.0)))
            self.summary.configure(text=f"Highest contribution: {leader['species']}  •  weight {float(leader['weight']):.3f}  •  compatibility {float(leader.get('compatibility', 0.0)):.3f}")
        else:
            self.summary.configure(text="")
        self.animation_progress = 0.0
        self._animate()

    def _animate(self) -> None:
        self.animation_progress = min(1.0, self.animation_progress + 0.09)
        self._draw_chart()
        if self.animation_progress < 1.0:
            self.anim_after = self.after(24, self._animate)
        else:
            self.anim_after = None

    def _draw_chart(self) -> None:
        self.canvas.delete("all")
        rows = self.report_rows
        width = max(320, self.canvas.winfo_width())
        height = max(220, self.canvas.winfo_height())
        if not rows:
            self.canvas.create_text(width / 2, height / 2, text="No weight preview yet", fill=MUTED, font=("Arial", 14))
            return
        left = 20
        label_w = 150
        bar_left = left + label_w
        right = width - 26
        usable = max(60, right - bar_left)
        top = 22
        row_h = max(36, min(56, (height - 24) // max(1, len(rows))))
        for idx, row in enumerate(rows):
            y0 = top + idx * row_h
            y1 = y0 + row_h - 12
            cy = (y0 + y1) / 2
            species = str(row.get("species", "unknown"))
            weight = float(row.get("weight", 0.0))
            compat = float(row.get("compatibility", 0.0))
            morph = float(row.get("morph_distance", 0.0))
            fill_w = usable * max(0.0, min(1.0, weight)) * self.animation_progress
            self.canvas.create_text(left, cy - 7, anchor="w", text=species, fill=TEXT, font=("Arial", 12, "bold"))
            self.canvas.create_text(left, cy + 10, anchor="w", text=f"compat {compat:.3f}   morph {morph:.3f}", fill=MUTED, font=("Arial", 10))
            self.canvas.create_rectangle(bar_left, y0, right, y1, fill="#243241", outline="#304150", width=1)
            self.canvas.create_rectangle(bar_left, y0, bar_left + fill_w, y1, fill=ACCENT, outline="")
            self.canvas.create_text(right - 8, cy, anchor="e", text=f"{weight:.3f}", fill=TEXT, font=("Arial", 11, "bold"))


class HeroCarousel:
    def __init__(self, label: ctk.CTkLabel, images: List[ctk.CTkImage], interval_ms: int = 2600) -> None:
        self.label = label
        self.images = images
        self.interval_ms = interval_ms
        self.idx = 0
        self.after_id: str | None = None

    def start(self) -> None:
        if self.images:
            self._tick()

    def _tick(self) -> None:
        self.label.configure(image=self.images[self.idx], text="")
        self.idx = (self.idx + 1) % len(self.images)
        self.after_id = self.label.after(self.interval_ms, self._tick)


class SpeciesPickerDialog(ctk.CTkToplevel):
    def __init__(self, parent: ctk.CTk, species_options: List[Dict[str, Any]], default_species: str | None) -> None:
        super().__init__(parent)
        self.title("Import bird motion clip")
        self.geometry("460x250")
        self.resizable(False, False)
        self.configure(fg_color=BG)
        self.result: Dict[str, Any] | None = None
        self.grab_set()

        self.species_var = ctk.StringVar(value=default_species or (species_options[0]["stem"] if species_options else ""))
        self.fps_var = ctk.StringVar(value="30")
        self.name_var = ctk.StringVar(value="")

        card = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=22, border_width=1, border_color=BORDER)
        card.pack(fill="both", expand=True, padx=18, pady=18)
        ctk.CTkLabel(card, text="Import a reference bird clip", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT).pack(anchor="w", padx=18, pady=(16, 6))
        ctk.CTkLabel(card, text="Assign the clip to a bird species profile and set playback FPS.", text_color=MUTED).pack(anchor="w", padx=18)

        form = ctk.CTkFrame(card, fg_color="transparent")
        form.pack(fill="x", padx=18, pady=14)
        form.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(form, text="Bird species", text_color=TEXT).grid(row=0, column=0, sticky="w", pady=6)
        ctk.CTkOptionMenu(form, variable=self.species_var, values=[item["stem"] for item in species_options] or [""], fg_color=CARD, button_color=ACCENT, button_hover_color=ACCENT_HOVER).grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=6)
        ctk.CTkLabel(form, text="Display name", text_color=TEXT).grid(row=1, column=0, sticky="w", pady=6)
        ctk.CTkEntry(form, textvariable=self.name_var, fg_color=CARD, border_color=BORDER).grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=6)
        ctk.CTkLabel(form, text="FPS", text_color=TEXT).grid(row=2, column=0, sticky="w", pady=6)
        ctk.CTkEntry(form, textvariable=self.fps_var, fg_color=CARD, border_color=BORDER).grid(row=2, column=1, sticky="ew", padx=(12, 0), pady=6)

        buttons = ctk.CTkFrame(card, fg_color="transparent")
        buttons.pack(fill="x", padx=18, pady=(0, 16))
        ctk.CTkButton(buttons, text="Cancel", fg_color="transparent", border_width=1, border_color=BORDER, text_color=TEXT, hover_color="#1A2430", command=self.destroy).pack(side="right")
        ctk.CTkButton(buttons, text="Import clip", fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#120B02", command=self._submit).pack(side="right", padx=(0, 10))

    def _submit(self) -> None:
        try:
            fps = float(self.fps_var.get())
        except ValueError:
            messagebox.showerror("Invalid FPS", "FPS must be numeric.", parent=self)
            return
        self.result = {
            "species_stem": self.species_var.get(),
            "display_name": self.name_var.get().strip() or None,
            "fps": fps,
        }
        self.destroy()


class JurassicGaitStudio(ctk.CTk):
    def __init__(self, start_page: str = "dashboard", screenshot_path: str | None = None, exit_after: int | None = None) -> None:
        super().__init__(fg_color=BG)
        bootstrap_workspace()
        self.title("Jurassic Gait Studio")
        self.geometry("1600x960")
        self.minsize(1180, 760)

        self.screenshot_path = screenshot_path
        self.exit_after = exit_after
        self.page_title_var = ctk.StringVar(value="Dashboard")
        self.status_var = ctk.StringVar(value="Ready")
        self.target_var = ctk.StringVar(value="")
        self.fps_var = ctk.StringVar(value="30")
        self.repeat_var = ctk.StringVar(value="3")
        self.phase_var = ctk.StringVar(value="101")
        self.lab_group_var = ctk.StringVar(value="theropod")
        self.lab_selected_stem: str | None = None
        self.current_page = ""
        self.asset_cache: Dict[str, ctk.CTkImage] = {}
        self.page_frames: Dict[str, ctk.CTkFrame] = {}
        self.sidebar_buttons: Dict[str, ctk.CTkButton] = {}
        self.selected_clip_ids: set[str] = set()
        self.latest_run_payload: Dict[str, Any] | None = None
        self.last_history_report_path: Path | None = None
        self.last_weight_preview_path = META_DIR / "last_weight_preview.png"
        self.lab_preview_path = META_DIR / "lab_species_preview.png"
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.generate_clip_vars: Dict[str, tk.BooleanVar] = {}
        self.library_clip_vars: Dict[str, tk.BooleanVar] = {}
        self.hero_carousel: HeroCarousel | None = None
        self.floating_viewers: list[Interactive3DViewer] = []
        self.lab_live_viewer: Interactive3DViewer | None = None
        self.lab_payload: Dict[str, Any] | None = None
        self.lab_drafts: Dict[str, Dict[str, Any]] = {}
        self.lab_quick_vars: Dict[str, ctk.StringVar] = {}
        self.lab_hint_labels: List[ctk.CTkLabel] = []
        self._resize_after_id: str | None = None

        self._build_shell()
        self._build_pages()
        self.refresh_all()
        self.show_page(start_page, instant=True)
        self.after(120, self._poll_worker_queue)
        if self.hero_carousel:
            self.hero_carousel.start()
        if self.screenshot_path:
            self.after(2000, self.capture_screenshot)
        if self.exit_after:
            self.after(self.exit_after, self.destroy)

    def _build_shell(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, fg_color="#0A0D11", width=300, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(8, weight=1)

        ctk.CTkLabel(sidebar, image=self._load_image("logo_amber.png", (116, 116)), text="").grid(row=0, column=0, sticky="w", padx=24, pady=(24, 8))
        ctk.CTkLabel(sidebar, text="Jurassic Gait Studio", font=ctk.CTkFont(size=28, weight="bold"), text_color=TEXT).grid(row=1, column=0, sticky="w", padx=24)
        ctk.CTkLabel(sidebar, text="A theropod gait reconstructing app.", text_color=MUTED, wraplength=238, justify="left").grid(row=2, column=0, sticky="w", padx=24, pady=(4, 18))

        nav_items = [
            ("dashboard", "Dashboard"),
            ("library", "Library"),
            ("lab", "Morphology Lab"),
            ("generate", "Generate"),
            ("history", "History"),
        ]
        for idx, (key, title) in enumerate(nav_items, start=3):
            btn = ctk.CTkButton(sidebar, text=title, anchor="w", corner_radius=16, height=46, fg_color="transparent", hover_color="#1A2430", text_color=TEXT, font=ctk.CTkFont(size=18, weight="bold"), command=lambda page=key: self.show_page(page))
            btn.grid(row=idx, column=0, sticky="ew", padx=18, pady=6)
            self.sidebar_buttons[key] = btn

        main = ctk.CTkFrame(self, fg_color=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)
        self.main = main

        header = ctk.CTkFrame(main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 10))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, textvariable=self.page_title_var, font=ctk.CTkFont(size=34, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, textvariable=self.status_var, text_color=MUTED).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.progress = ctk.CTkProgressBar(header, width=240, progress_color=ACCENT)
        self.progress.grid(row=0, column=1, rowspan=2, sticky="e")
        self.progress.set(0)

        self.content = ctk.CTkFrame(main, fg_color=BG)
        self.content.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 18))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)
        self.bind("<Configure>", self._on_app_resize, add="+")

    def _build_pages(self) -> None:
        self.page_frames = {
            "dashboard": self._build_dashboard_page(),
            "library": self._build_library_page(),
            "lab": self._build_lab_page(),
            "generate": self._build_generate_page(),
            "history": self._build_history_page(),
        }
        for frame in self.page_frames.values():
            frame.place(in_=self.content, x=0, y=0, relwidth=1, relheight=1)
            frame.place_forget()
        self.after(80, self._apply_responsive_layouts)

    def _build_dashboard_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content, fg_color=BG)
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(3, weight=1)

        hero = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=28, border_width=1, border_color=BORDER)
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        hero.grid_columnconfigure(1, weight=1)
        self.hero_label = ctk.CTkLabel(hero, text="")
        self.hero_label.grid(row=0, column=0, padx=18, pady=18)
        self.hero_carousel = HeroCarousel(self.hero_label, [self._load_image("banner_gate.png", (450, 250)), self._load_image("banner_dilo.png", (450, 250)), self._load_image("banner_brachio.png", (450, 250))])
        info = ctk.CTkFrame(hero, fg_color="transparent")
        info.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=18)
        ctk.CTkLabel(info, text="Design, edit, and animate theropod locomotion.", font=ctk.CTkFont(size=30, weight="bold"), text_color=TEXT, justify="left", wraplength=700).pack(anchor="w", pady=(4, 12))
        ctk.CTkLabel(info, text="Import bird CSV clips from SLEAP, manage bird and dinosaur JSONs, edit morphology, preview body shape, generate numbered runs, and inspect results in both GIF and interactive 3D views.", text_color=MUTED, justify="left", wraplength=720).pack(anchor="w")
        actions = ctk.CTkFrame(info, fg_color="transparent")
        actions.pack(anchor="w", pady=(18, 0))
        ctk.CTkButton(actions, text="Import bird CSV", fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#120B02", command=self.import_clip_flow).pack(side="left", padx=(0, 10))
        ctk.CTkButton(actions, text="Open Morphology Lab", fg_color=VIOLET, hover_color=VIOLET_HOVER, command=lambda: self.show_page("lab")).pack(side="left", padx=(0, 10))
        ctk.CTkButton(actions, text="Generate gait", fg_color=INFO, hover_color="#4D7493", command=lambda: self.show_page("generate")).pack(side="left")

        self.dashboard_stats = ctk.CTkFrame(page, fg_color="transparent")
        self.dashboard_stats.grid(row=1, column=0, sticky="ew", pady=(0, 18))
        for i in range(4):
            self.dashboard_stats.grid_columnconfigure(i, weight=1)

        workflow = ctk.CTkFrame(page, fg_color="transparent")
        workflow.grid(row=2, column=0, sticky="ew", pady=(0, 18))
        for i, (title, subtitle, cb) in enumerate([
            ("1. Fill the library", "Import bird reference CSVs and JSON profiles.", lambda: self.show_page("library")),
            ("2. Tune species JSON", "Edit bird or theropod proportions and preview the body shape before saving.", lambda: self.show_page("lab")),
            ("3. Generate + inspect", "Preview fusion weights, save numbered runs, and open interactive 3D playback.", lambda: self.show_page("generate")),
        ]):
            workflow.grid_columnconfigure(i, weight=1)
            card = ctk.CTkFrame(workflow, fg_color=CARD, corner_radius=22, border_width=1, border_color=BORDER)
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 8 if i < 2 else 0))
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).pack(anchor="w", padx=16, pady=(16, 6))
            ctk.CTkLabel(card, text=subtitle, wraplength=360, justify="left", text_color=MUTED).pack(anchor="w", padx=16, pady=(0, 10))
            ctk.CTkButton(card, text="Open", fg_color=INFO, hover_color="#4D7493", command=cb).pack(anchor="w", padx=16, pady=(0, 16))

        recent = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=24, border_width=1, border_color=BORDER)
        recent.grid(row=3, column=0, sticky="nsew")
        recent.grid_columnconfigure(0, weight=1)
        recent.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(recent, text="Recent runs", font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 10))
        self.dashboard_recent_runs = ctk.CTkScrollableFrame(recent, fg_color="transparent")
        self.dashboard_recent_runs.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        return page

    def _build_library_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content, fg_color=BG)
        page.grid_columnconfigure((0, 1, 2), weight=1)
        page.grid_rowconfigure(1, weight=1)

        actions = ctk.CTkFrame(page, fg_color="transparent")
        actions.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 14))
        for i in range(3):
            actions.grid_columnconfigure(i, weight=1)
        for col, (title, subtitle, color, cb, dark) in enumerate([
            ("Import bird JSON", "Add a new modern bird species profile.", ACCENT_HOVER, self.import_bird_species_flow, False),
            ("Import bird CSV", "Bring in a new SLEAP or legacy motion clip.", ACCENT, self.import_clip_flow, True),
            ("Import target JSON", "Add a new theropod target species.", VIOLET, self.import_target_species_flow, False),
        ]):
            card = ctk.CTkFrame(actions, fg_color=PANEL, corner_radius=22, border_width=1, border_color=BORDER)
            card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 8 if col < 2 else 0))
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).pack(anchor="w", padx=16, pady=(16, 6))
            ctk.CTkLabel(card, text=subtitle, wraplength=320, justify="left", text_color=MUTED).pack(anchor="w", padx=16, pady=(0, 12))
            ctk.CTkButton(card, text="Open", fg_color=color, hover_color=ACCENT_HOVER if color == ACCENT else VIOLET_HOVER if color == VIOLET else "#F6CC79", text_color="#120B02" if dark or color == ACCENT else TEXT, command=cb).pack(anchor="w", padx=16, pady=(0, 16))

        self.library_birds_list = self._library_panel(page, 1, 0, "Bird species")
        self.library_targets_list = self._library_panel(page, 1, 1, "Target theropods")
        self.library_clips_list = self._library_panel(page, 1, 2, "Reference clips")
        return page

    def _library_panel(self, parent: ctk.CTkFrame, row: int, col: int, title: str) -> ctk.CTkScrollableFrame:
        panel = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=24, border_width=1, border_color=BORDER)
        panel.grid(row=row, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0 if col == 2 else 10))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(panel, text=title, font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))
        scroll = ctk.CTkScrollableFrame(panel, fg_color="transparent")
        scroll.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        return scroll

    def _build_lab_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content, fg_color=BG)
        page.grid_columnconfigure(0, weight=3, minsize=280, uniform="labcols")
        page.grid_columnconfigure(1, weight=5, minsize=520, uniform="labcols")
        page.grid_columnconfigure(2, weight=4, minsize=360, uniform="labcols")
        page.grid_rowconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        left = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=24, border_width=1, border_color=BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(left, text="Morphology Lab", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))
        self.lab_group_switch = ctk.CTkSegmentedButton(left, values=["theropod", "bird"], variable=self.lab_group_var, command=lambda _v: self._refresh_lab_list())
        self.lab_group_switch.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))
        self.lab_species_list = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.lab_species_list.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))

        center = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=24, border_width=1, border_color=BORDER)
        center.grid(row=0, column=1, sticky="nsew", padx=12)
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(2, weight=1)
        controls_head = ctk.CTkFrame(center, fg_color="transparent")
        controls_head.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))
        controls_head.grid_columnconfigure(0, weight=1)
        title_col = ctk.CTkFrame(controls_head, fg_color="transparent")
        title_col.grid(row=0, column=0, sticky="ew")
        title_col.grid_columnconfigure(0, weight=1)
        self.lab_controls_title = ctk.CTkLabel(title_col, text="Simplified morphology controls", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT)
        self.lab_controls_title.grid(row=0, column=0, sticky="w")
        self.lab_controls_subtitle = ctk.CTkLabel(title_col, text="Direct rig controls only: lengths, diameters, and angles.", text_color=MUTED, justify="left", wraplength=520)
        self.lab_controls_subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))
        btns = ctk.CTkFrame(controls_head, fg_color="transparent")
        btns.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        btns.grid_columnconfigure((0, 1), weight=1)
        self.lab_apply_button = ctk.CTkButton(btns, text="Apply to preview", fg_color=INFO, hover_color="#4D7493", command=self.lab_apply_quick_controls)
        self.lab_apply_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.lab_reset_button = ctk.CTkButton(btns, text="Reset", fg_color="#2B3440", hover_color="#3C4958", command=self.lab_reload_controls_from_editor)
        self.lab_reset_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        path_bar = ctk.CTkFrame(center, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER)
        path_bar.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))
        path_bar.grid_columnconfigure(0, weight=1)
        self.lab_path_label = ctk.CTkLabel(path_bar, text="Species file: select one", text_color=MUTED, anchor="w", justify="left")
        self.lab_path_label.grid(row=0, column=0, sticky="ew", padx=12, pady=10)

        self.lab_controls_scroll = ctk.CTkScrollableFrame(center, fg_color="transparent")
        self.lab_controls_scroll.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self._build_lab_quick_controls(self.lab_controls_scroll)

        right = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=24, border_width=1, border_color=BORDER)
        right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(right, text="Shape preview", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))
        self.lab_preview_status = ctk.CTkLabel(right, text="Pick a species, tweak controls, then apply.", text_color=MUTED, justify="left")
        self.lab_preview_status.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 8))

        preview_frame = ctk.CTkFrame(right, fg_color="#10161D", corner_radius=18, border_width=1, border_color=BORDER)
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 10))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)
        self.lab_preview_label = ctk.CTkLabel(preview_frame, text="Apply changes to visualize the current model.", text_color=MUTED)
        self.lab_preview_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.lab_preview_player = ImagePlayer(self.lab_preview_label, (420, 320), fill_bg="#10161D")

        self.lab_summary = ctk.CTkTextbox(right, height=148, fg_color="#10161D", border_color=BORDER, border_width=1, text_color=TEXT)
        self.lab_summary.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 10))
        preview_buttons = ctk.CTkFrame(right, fg_color="transparent")
        preview_buttons.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 18))
        preview_buttons.grid_columnconfigure((0, 1), weight=1)
        self.lab_preview_apply_button = ctk.CTkButton(preview_buttons, text="Apply", fg_color=INFO, hover_color="#4D7493", command=self.lab_apply_quick_controls)
        self.lab_preview_apply_button.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))
        self.lab_preview_open3d_button = ctk.CTkButton(preview_buttons, text="Open 3D", fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#120B02", command=self.lab_open_3d_preview)
        self.lab_preview_open3d_button.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=(0, 8))
        self.lab_preview_save_button = ctk.CTkButton(preview_buttons, text="Save", fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#120B02", command=self.lab_save_current)
        self.lab_preview_save_button.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.lab_preview_save_as_button = ctk.CTkButton(preview_buttons, text="Save as new", fg_color=VIOLET, hover_color=VIOLET_HOVER, command=self.lab_save_as_new)
        self.lab_preview_save_as_button.grid(row=1, column=1, sticky="ew", padx=(6, 0))
        self.lab_page = page
        self.lab_left_panel = left
        self.lab_center_panel = center
        self.lab_right_panel = right
        return page

    def _build_generate_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content, fg_color=BG)
        page.grid_columnconfigure(0, weight=3, minsize=360, uniform="gencols")
        page.grid_columnconfigure(1, weight=5, minsize=540, uniform="gencols")
        page.grid_rowconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        controls = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=24, border_width=1, border_color=BORDER)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_rowconfigure(5, weight=1)
        ctk.CTkLabel(controls, text="Generate a new run", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))
        strip = ctk.CTkFrame(controls, fg_color=CARD, corner_radius=18, border_width=1, border_color=BORDER)
        strip.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 12))
        strip.grid_columnconfigure(0, weight=1)
        self.target_menu = ctk.CTkOptionMenu(strip, variable=self.target_var, values=[""], command=self._select_target, fg_color=CARD_ALT, button_color=VIOLET, button_hover_color=VIOLET_HOVER)
        self.target_menu.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        self.generate_target_info = ctk.CTkLabel(strip, text="Select a target dinosaur.", text_color=MUTED, justify="left", wraplength=330)
        self.generate_target_info.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 12))

        params = ctk.CTkFrame(controls, fg_color=CARD, corner_radius=18, border_width=1, border_color=BORDER)
        params.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        params.grid_columnconfigure(1, weight=1)
        for idx, (label, var) in enumerate([("FPS", self.fps_var), ("Repeat cycles", self.repeat_var), ("Phase frames", self.phase_var)]):
            ctk.CTkLabel(params, text=label, text_color=TEXT).grid(row=idx, column=0, sticky="w", padx=14, pady=8)
            ctk.CTkEntry(params, textvariable=var, fg_color="#10161D", border_color=BORDER).grid(row=idx, column=1, sticky="ew", padx=(0, 14), pady=8)

        clip_toolbar = ctk.CTkFrame(controls, fg_color="transparent")
        clip_toolbar.grid(row=3, column=0, sticky="ew", padx=18)
        clip_toolbar.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(clip_toolbar, text="Reference bird clips", text_color=TEXT).grid(row=0, column=0, sticky="w")
        clip_btns = ctk.CTkFrame(clip_toolbar, fg_color="transparent")
        clip_btns.grid(row=0, column=1, sticky="e")
        ctk.CTkButton(clip_btns, text="All", width=64, fg_color=INFO, hover_color="#4D7493", command=self.select_all_clips).pack(side="left", padx=(8, 0))
        ctk.CTkButton(clip_btns, text="None", width=64, fg_color="#2B3440", hover_color="#3C4958", command=self.clear_all_clips).pack(side="left")
        self.generate_clip_count = ctk.CTkLabel(controls, text="", text_color=MUTED)
        self.generate_clip_count.grid(row=4, column=0, sticky="w", padx=18, pady=(4, 6))
        self.generate_clips_frame = ctk.CTkScrollableFrame(controls, fg_color="transparent")
        self.generate_clips_frame.grid(row=5, column=0, sticky="nsew", padx=14, pady=(0, 14))

        actions = ctk.CTkFrame(controls, fg_color="transparent")
        actions.grid(row=6, column=0, sticky="ew", padx=18, pady=(0, 18))
        actions.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkButton(actions, text="Preview weights", fg_color=INFO, hover_color="#4D7493", command=self.preview_weights_flow).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(actions, text="Generate gait", fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#120B02", command=self.generate_run_flow).grid(row=0, column=1, sticky="ew", padx=6)
        ctk.CTkButton(actions, text="Open 3D viewer", fg_color="transparent", border_width=1, border_color=BORDER, hover_color="#1A2430", command=self.open_latest_run_3d_viewer).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        out = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=24, border_width=1, border_color=BORDER)
        out.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        out.grid_columnconfigure(0, weight=1)
        out.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(out, text="Run preview + diagnostics", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))
        self.generate_run_badge = ctk.CTkLabel(out, text="No run yet", text_color=MUTED)
        self.generate_run_badge.grid(row=0, column=0, sticky="e", padx=18)
        self.generate_tabs = ctk.CTkTabview(out, fg_color="transparent", segmented_button_fg_color=CARD_ALT, segmented_button_selected_color=ACCENT, segmented_button_selected_hover_color=ACCENT_HOVER, segmented_button_unselected_color=CARD)
        self.generate_tabs.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        for tab in ["Preview", "Fusion", "Session", "Files"]:
            self.generate_tabs.add(tab)

        preview_tab = self.generate_tabs.tab("Preview")
        preview_tab.grid_columnconfigure(0, weight=1)
        preview_tab.grid_rowconfigure(1, weight=1)
        preview_actions = ctk.CTkFrame(preview_tab, fg_color="transparent")
        preview_actions.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 6))
        preview_actions.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(preview_actions, text="Export GIF", fg_color=INFO, hover_color="#4D7493", command=self.export_latest_gif).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(preview_actions, text="Open 3D viewer", fg_color="transparent", border_width=1, border_color=BORDER, hover_color="#1A2430", command=self.open_latest_run_3d_viewer).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        preview_frame = ctk.CTkFrame(preview_tab, fg_color="#10161D", corner_radius=18, border_width=1, border_color=BORDER)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)
        self.generate_preview_label = ctk.CTkLabel(preview_frame, text="Generate a target to preview the GIF here.", text_color=MUTED)
        self.generate_preview_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.generate_preview_player = ImagePlayer(self.generate_preview_label, (760, 360), fill_bg="#10161D")

        fusion_tab = self.generate_tabs.tab("Fusion")
        fusion_tab.grid_columnconfigure(0, weight=1)
        fusion_tab.grid_rowconfigure(1, weight=1)
        self.generate_weight_chart = FusionBarsPanel(fusion_tab)
        self.generate_weight_chart.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 8))
        self.generate_fusion_box = ctk.CTkTextbox(fusion_tab, fg_color="#10161D", border_color=BORDER, border_width=1, text_color=TEXT)
        self.generate_fusion_box.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        self.generate_session_box = ctk.CTkTextbox(self.generate_tabs.tab("Session"), fg_color="#10161D", border_color=BORDER, border_width=1, text_color=TEXT)
        self.generate_session_box.pack(fill="both", expand=True, padx=6, pady=6)
        self.generate_files_box = ctk.CTkTextbox(self.generate_tabs.tab("Files"), fg_color="#10161D", border_color=BORDER, border_width=1, text_color=TEXT)
        self.generate_files_box.pack(fill="both", expand=True, padx=6, pady=6)
        self.generate_page = page
        self.generate_controls_panel = controls
        self.generate_output_panel = out
        return page

    def _build_history_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content, fg_color=BG)
        page.grid_columnconfigure(0, weight=4, minsize=360, uniform="histcols")
        page.grid_columnconfigure(1, weight=5, minsize=540, uniform="histcols")
        page.grid_rowconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        left = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=24, border_width=1, border_color=BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(left, text="Saved run history", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))
        self.history_list = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.history_list.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        right = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=24, border_width=1, border_color=BORDER)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(right, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))
        head.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkLabel(head, text="Selected run details", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(head, text="Export GIF", fg_color=INFO, hover_color="#4D7493", command=self.export_history_gif).grid(row=0, column=1, sticky="ew", padx=6)
        ctk.CTkButton(head, text="Open 3D viewer", fg_color="transparent", border_width=1, border_color=BORDER, hover_color="#1A2430", command=self.open_history_run_3d_viewer).grid(row=0, column=2, sticky="ew")
        self.history_tabs = ctk.CTkTabview(right, fg_color="transparent", segmented_button_fg_color=CARD_ALT, segmented_button_selected_color=VIOLET, segmented_button_selected_hover_color=VIOLET_HOVER, segmented_button_unselected_color=CARD)
        self.history_tabs.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        for tab in ["Preview", "Report", "Raw JSON", "Files"]:
            self.history_tabs.add(tab)
        preview_tab = self.history_tabs.tab("Preview")
        preview_tab.grid_columnconfigure(0, weight=1)
        preview_tab.grid_rowconfigure(0, weight=1)
        preview_frame = ctk.CTkFrame(preview_tab, fg_color="#10161D", corner_radius=18, border_width=1, border_color=BORDER)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)
        self.history_preview_label = ctk.CTkLabel(preview_frame, text="Pick a run to preview the saved GIF.", text_color=MUTED)
        self.history_preview_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.history_preview_player = ImagePlayer(self.history_preview_label, (760, 360), fill_bg="#10161D")
        self.history_report_box = ctk.CTkTextbox(self.history_tabs.tab("Report"), fg_color="#10161D", border_color=BORDER, border_width=1, text_color=TEXT)
        self.history_report_box.pack(fill="both", expand=True, padx=6, pady=6)
        self.history_json_box = ctk.CTkTextbox(self.history_tabs.tab("Raw JSON"), fg_color="#10161D", border_color=BORDER, border_width=1, text_color=TEXT)
        self.history_json_box.pack(fill="both", expand=True, padx=6, pady=6)
        self.history_files_box = ctk.CTkTextbox(self.history_tabs.tab("Files"), fg_color="#10161D", border_color=BORDER, border_width=1, text_color=TEXT)
        self.history_files_box.pack(fill="both", expand=True, padx=6, pady=6)
        self.history_page = page
        self.history_left_panel = left
        self.history_right_panel = right
        return page

    def _on_app_resize(self, _event=None) -> None:
        if self._resize_after_id:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.after(90, self._apply_responsive_layouts)

    def _apply_responsive_layouts(self) -> None:
        self._resize_after_id = None
        try:
            self._layout_lab_page()
            self._layout_generate_page()
            self._layout_history_page()
        except Exception:
            pass

    def _layout_lab_page(self) -> None:
        page = getattr(self, "lab_page", None)
        if page is None:
            return
        width = max(page.winfo_width(), self.content.winfo_width())
        left = self.lab_left_panel
        center = self.lab_center_panel
        right = self.lab_right_panel
        center_width = max(center.winfo_width(), width // 2)
        if width >= 1500:
            page.grid_columnconfigure(0, weight=3, minsize=280, uniform="labcols")
            page.grid_columnconfigure(1, weight=5, minsize=520, uniform="labcols")
            page.grid_columnconfigure(2, weight=4, minsize=360, uniform="labcols")
            left.grid_configure(row=0, column=0, rowspan=1, columnspan=1, padx=(0, 12), pady=0, sticky="nsew")
            center.grid_configure(row=0, column=1, rowspan=1, columnspan=1, padx=12, pady=0, sticky="nsew")
            right.grid_configure(row=0, column=2, rowspan=1, columnspan=1, padx=(12, 0), pady=0, sticky="nsew")
            if hasattr(self, 'lab_controls_subtitle'):
                self.lab_controls_subtitle.configure(text="Direct rig controls only: lengths, diameters, and angles.", wraplength=520)
        elif width >= 1120:
            page.grid_columnconfigure(0, weight=3, minsize=260, uniform="labstack")
            page.grid_columnconfigure(1, weight=7, minsize=540, uniform="labstack")
            page.grid_columnconfigure(2, weight=0, minsize=0)
            left.grid_configure(row=0, column=0, rowspan=2, columnspan=1, padx=(0, 12), pady=0, sticky="nsew")
            center.grid_configure(row=0, column=1, rowspan=1, columnspan=1, padx=(12, 0), pady=0, sticky="nsew")
            right.grid_configure(row=1, column=1, rowspan=1, columnspan=1, padx=(12, 0), pady=(12, 0), sticky="nsew")
            if hasattr(self, 'lab_controls_subtitle'):
                self.lab_controls_subtitle.configure(text="Direct controls: length, diameter, angle.", wraplength=420)
        else:
            page.grid_columnconfigure(0, weight=1, minsize=0)
            page.grid_columnconfigure(1, weight=0, minsize=0)
            page.grid_columnconfigure(2, weight=0, minsize=0)
            left.grid_configure(row=0, column=0, rowspan=1, columnspan=1, padx=0, pady=(0, 12), sticky="nsew")
            center.grid_configure(row=1, column=0, rowspan=1, columnspan=1, padx=0, pady=(0, 12), sticky="nsew")
            right.grid_configure(row=2, column=0, rowspan=1, columnspan=1, padx=0, pady=0, sticky="nsew")
            if hasattr(self, 'lab_controls_subtitle'):
                self.lab_controls_subtitle.configure(text="Length • diameter • angle", wraplength=280)
        hint_wrap = 320 if center_width >= 760 else 220 if center_width >= 560 else 170
        for label in getattr(self, 'lab_hint_labels', []):
            try:
                label.configure(wraplength=hint_wrap)
            except Exception:
                pass

    def _layout_generate_page(self) -> None:
        page = getattr(self, "generate_page", None)
        if page is None:
            return
        width = max(page.winfo_width(), self.content.winfo_width())
        left = self.generate_controls_panel
        right = self.generate_output_panel
        if width >= 1180:
            page.grid_columnconfigure(0, weight=3, minsize=360, uniform="gencols")
            page.grid_columnconfigure(1, weight=5, minsize=560, uniform="gencols")
            left.grid_configure(row=0, column=0, columnspan=1, padx=(0, 12), pady=0, sticky="nsew")
            right.grid_configure(row=0, column=1, columnspan=1, padx=(12, 0), pady=0, sticky="nsew")
        else:
            page.grid_columnconfigure(0, weight=1, minsize=0)
            page.grid_columnconfigure(1, weight=0, minsize=0)
            left.grid_configure(row=0, column=0, columnspan=1, padx=0, pady=(0, 12), sticky="nsew")
            right.grid_configure(row=1, column=0, columnspan=1, padx=0, pady=0, sticky="nsew")

    def _layout_history_page(self) -> None:
        page = getattr(self, "history_page", None)
        if page is None:
            return
        width = max(page.winfo_width(), self.content.winfo_width())
        left = self.history_left_panel
        right = self.history_right_panel
        if width >= 1180:
            page.grid_columnconfigure(0, weight=4, minsize=360, uniform="histcols")
            page.grid_columnconfigure(1, weight=5, minsize=560, uniform="histcols")
            left.grid_configure(row=0, column=0, columnspan=1, padx=(0, 12), pady=0, sticky="nsew")
            right.grid_configure(row=0, column=1, columnspan=1, padx=(12, 0), pady=0, sticky="nsew")
        else:
            page.grid_columnconfigure(0, weight=1, minsize=0)
            page.grid_columnconfigure(1, weight=0, minsize=0)
            left.grid_configure(row=0, column=0, columnspan=1, padx=0, pady=(0, 12), sticky="nsew")
            right.grid_configure(row=1, column=0, columnspan=1, padx=0, pady=0, sticky="nsew")

    def _load_image(self, name: str, size: tuple[int, int]) -> ctk.CTkImage:
        key = f"{name}:{size[0]}x{size[1]}"
        if key in self.asset_cache:
            return self.asset_cache[key]
        image = Image.open(ASSETS_DIR / name).convert("RGBA")
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=size)
        self.asset_cache[key] = ctk_image
        return ctk_image

    def _lab_draft_key(self, group: str, stem: str | None) -> str | None:
        return f"{group}:{stem}" if stem else None

    def _get_lab_draft(self, group: str, stem: str | None) -> Dict[str, Any] | None:
        key = self._lab_draft_key(group, stem)
        if key is None:
            return None
        payload = self.lab_drafts.get(key)
        return json.loads(json.dumps(payload)) if payload is not None else None

    def _set_lab_draft(self, group: str, stem: str | None, payload: Dict[str, Any]) -> None:
        key = self._lab_draft_key(group, stem)
        if key is None:
            return
        self.lab_drafts[key] = json.loads(json.dumps(payload))

    def _clear_lab_draft(self, group: str, stem: str | None) -> None:
        key = self._lab_draft_key(group, stem)
        if key is not None:
            self.lab_drafts.pop(key, None)

    def _collect_generation_overrides(self) -> tuple[Dict[str, Any] | None, Dict[str, Dict[str, Any]]]:
        target_override = self._get_lab_draft("theropod", self.target_var.get())
        bird_overrides: Dict[str, Dict[str, Any]] = {}
        selected = {cid for cid in self.selected_clip_ids}
        for clip in self.bird_clips:
            if clip["clip_id"] not in selected:
                continue
            stem = str(clip.get("species_stem") or "")
            draft = self._get_lab_draft("bird", stem)
            if draft is not None:
                bird_overrides[stem] = draft
        return target_override, bird_overrides

    def _selected_target_summary(self) -> Dict[str, Any] | None:
        stem = self.target_var.get()
        if not stem:
            return None
        draft = self._get_lab_draft("theropod", stem)
        if draft is not None:
            species = species_from_payload(draft)
            return {"stem": stem, "name": species.name, "height_m": species.height_m, "mass_kg": species.mass_kg, "draft": True}
        return next((item for item in self.targets if item["stem"] == stem), None)

    def refresh_all(self) -> None:
        self.bird_species = list_species("bird")
        self.targets = list_species("theropod")
        self.bird_clips = list_bird_clips()
        self.runs = list_runs(limit=50)
        if self.latest_run_payload is None and self.runs:
            report_path = Path(self.runs[0]["report_path"])
            self.latest_run_payload = {"run_dir": str(report_path.parent), "session_report": json.loads(report_path.read_text(encoding="utf-8"))}
        current_ids = {item["clip_id"] for item in self.bird_clips}
        self.selected_clip_ids = self.selected_clip_ids & current_ids if self.selected_clip_ids else set(current_ids)
        if not self.selected_clip_ids:
            self.selected_clip_ids = set(current_ids)
        if (not self.target_var.get()) or all(item["stem"] != self.target_var.get() for item in self.targets):
            default_target = "ornithomimid_template" if any(t["stem"] == "ornithomimid_template" for t in self.targets) else (self.targets[0]["stem"] if self.targets else "")
            self.target_var.set(default_target)
        self.selected_target_details = self._selected_target_summary()
        self._refresh_dashboard()
        self._refresh_library()
        self._refresh_lab_list()
        self._refresh_generate()
        self._refresh_history()

    def _refresh_dashboard(self) -> None:
        self._clear(self.dashboard_stats)
        for idx, (title, value, subtitle) in enumerate([
            ("Bird species", str(len(self.bird_species)), "Reference morphology JSON profiles"),
            ("Bird clips", str(len(self.bird_clips)), "Motion samples in database/bird_clips"),
            ("Target theropods", str(len(self.targets)), "Dinosaurs users can reconstruct"),
            ("Generated runs", str(len(self.runs)), "Sequential outputs in runs/<target>/run_###"),
        ]):
            self.dashboard_stats.grid_columnconfigure(idx, weight=1)
            card = ctk.CTkFrame(self.dashboard_stats, fg_color=CARD_ALT, corner_radius=22, border_width=1, border_color=BORDER)
            card.grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 8, 8 if idx < 3 else 0))
            ctk.CTkLabel(card, text=title, text_color=MUTED).pack(anchor="w", padx=16, pady=(14, 4))
            ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=34, weight="bold"), text_color=TEXT).pack(anchor="w", padx=16)
            ctk.CTkLabel(card, text=subtitle, wraplength=250, justify="left", text_color=MUTED).pack(anchor="w", padx=16, pady=(0, 14))
        self._clear(self.dashboard_recent_runs)
        if not self.runs:
            ctk.CTkLabel(self.dashboard_recent_runs, text="No runs yet.", text_color=MUTED).pack(anchor="w", padx=10, pady=10)
            return
        for run in self.runs[:8]:
            self._run_card(self.dashboard_recent_runs, run, compact=True).pack(fill="x", padx=6, pady=6)

    def _refresh_library(self) -> None:
        self._clear(self.library_birds_list)
        for species in self.bird_species:
            self._species_card(self.library_birds_list, species).pack(fill="x", padx=6, pady=6)
        self._clear(self.library_targets_list)
        for species in self.targets:
            self._species_card(self.library_targets_list, species).pack(fill="x", padx=6, pady=6)
        self._clear(self.library_clips_list)
        for clip in self.bird_clips:
            card = ctk.CTkFrame(self.library_clips_list, fg_color=CARD, corner_radius=20, border_width=1, border_color=BORDER)
            card.pack(fill="x", padx=6, pady=6)
            ctk.CTkLabel(card, text=clip["name"], font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).pack(anchor="w", padx=14, pady=(12, 2))
            ctk.CTkLabel(card, text=f"{clip['species_name']} • {clip['fps']:.1f} FPS", text_color=MUTED).pack(anchor="w", padx=14)
            ctk.CTkLabel(card, text=clip["csv_path"], text_color=MUTED, justify="left").pack(anchor="w", padx=14, pady=(0, 12))

    def _species_card(self, parent: ctk.CTkScrollableFrame, species: Dict[str, Any]) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=20, border_width=1, border_color=BORDER)
        ctk.CTkLabel(card, text=species["name"], font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).pack(anchor="w", padx=14, pady=(12, 2))
        ctk.CTkLabel(card, text=f"stem: {species['stem']}  •  height {species['height_m']:.2f} m  •  mass {species['mass_kg']:.1f} kg", text_color=MUTED).pack(anchor="w", padx=14, pady=(0, 12))
        return card

    def _refresh_lab_list(self) -> None:
        items = self.targets if self.lab_group_var.get() == "theropod" else self.bird_species
        self._clear(self.lab_species_list)
        if items and (self.lab_selected_stem not in {i['stem'] for i in items}):
            self.lab_selected_stem = items[0]["stem"]
        for item in items:
            selected = item["stem"] == self.lab_selected_stem
            btn = ctk.CTkButton(self.lab_species_list, text=f"{item['name']}", anchor="w", height=74, fg_color=VIOLET if selected else CARD, hover_color=VIOLET_HOVER if selected else CARD_HOVER, text_color=TEXT, corner_radius=18, command=lambda stem=item['stem']: self.lab_select_species(stem))
            btn.pack(fill="x", padx=6, pady=6)
        if items and self.lab_selected_stem:
            self.lab_load_species(self.lab_group_var.get(), self.lab_selected_stem)

    def lab_select_species(self, stem: str) -> None:
        self.lab_selected_stem = stem
        self._refresh_lab_list()

    def _build_lab_quick_controls(self, parent: ctk.CTkScrollableFrame) -> None:
        self.lab_quick_vars = {}
        self.lab_hint_labels = []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for spec in LAB_QUICK_FIELDS:
            grouped.setdefault(spec["group"], []).append(spec)
        for group_name, specs in grouped.items():
            card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=18, border_width=1, border_color=BORDER)
            card.pack(fill="x", padx=6, pady=8)
            card.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(card, text=group_name, font=ctk.CTkFont(size=19, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 8), columnspan=2)
            for idx, spec in enumerate(specs, start=1):
                row = ctk.CTkFrame(card, fg_color="transparent")
                row.grid(row=idx, column=0, sticky="ew", padx=12, pady=4)
                row.grid_columnconfigure(0, weight=1)
                text_box = ctk.CTkFrame(row, fg_color="transparent")
                text_box.grid(row=0, column=0, sticky="ew")
                ctk.CTkLabel(text_box, text=spec["label"], text_color=TEXT, anchor="w").pack(anchor="w")
                hint_label = ctk.CTkLabel(text_box, text=spec["hint"], text_color=MUTED, anchor="w", justify="left", wraplength=300)
                hint_label.pack(anchor="w")
                self.lab_hint_labels.append(hint_label)
                var = ctk.StringVar(value="")
                self.lab_quick_vars[spec["id"]] = var
                entry = ctk.CTkEntry(row, textvariable=var, width=120, fg_color=CARD_ALT, border_color=BORDER, justify="right")
                entry.grid(row=0, column=1, sticky="e", padx=(10, 0), pady=4)

    def _control_value_from_payload(self, payload: Dict[str, Any], field_id: str) -> float | str:
        species = species_from_payload(payload)
        if field_id == "hip_width_m":
            return species.hip_width
        if field_id == "rear_body_length_m":
            return species.rear_body_length
        if field_id == "front_body_length_m":
            return species.torso
        if field_id == "tail_length_m":
            return species.tail
        if field_id == "neck_length_m":
            return species.neck
        if field_id == "head_length_m":
            return species.head
        if field_id == "thigh_length_m":
            return species.thigh
        if field_id == "shank_length_m":
            return species.shank
        if field_id == "metatarsus_length_m":
            return species.metatarsus
        if field_id == "foot_length_m":
            return species.foot

        if field_id == "rear_body_diameter_m":
            return species.effective_rear_body_diameter_m
        if field_id == "front_body_diameter_m":
            return species.effective_front_body_diameter_m
        if field_id == "tail_base_diameter_m":
            return species.effective_tail_base_diameter_m
        if field_id == "tail_tip_diameter_m":
            return species.effective_tail_tip_diameter_m
        if field_id == "neck_diameter_m":
            return species.effective_neck_diameter_m
        if field_id == "head_diameter_m":
            return species.effective_head_diameter_m
        if field_id == "thigh_diameter_m":
            return species.effective_thigh_diameter_m
        if field_id == "shank_diameter_m":
            return species.effective_shank_diameter_m
        if field_id == "metatarsus_diameter_m":
            return species.effective_metatarsus_diameter_m

        if field_id == "rear_body_angle_deg":
            return species.effective_rear_body_angle_deg
        if field_id == "front_body_angle_deg":
            return float(payload.get("trunk_pitch_bias_deg", 0.0))
        if field_id == "tail_angle_deg":
            return float(payload.get("tail_pitch_deg", 0.0))
        if field_id == "neck_angle_deg":
            return species.effective_neck_angle_deg
        if field_id == "head_angle_deg":
            return species.effective_head_angle_deg
        return ""

    def _set_nested_float(self, payload: Dict[str, Any], path: tuple[str, ...], value: float) -> None:
        cur = payload
        for key in path[:-1]:
            nxt = cur.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[key] = nxt
            cur = nxt
        cur[path[-1]] = float(value)

    def _apply_control_to_payload(self, payload: Dict[str, Any], field_id: str, value: float) -> None:
        h = max(0.01, float(payload.get("height_m", 1.0)))
        if field_id == "hip_width_m":
            payload["hip_width_ratio"] = float(value) / h
            return
        if field_id == "rear_body_length_m":
            payload["pelvis_length_ratio"] = float(value) / h
            return
        if field_id == "front_body_length_m":
            self._set_nested_float(payload, ("limb_lengths_m", "torso"), value)
            return
        if field_id == "tail_length_m":
            self._set_nested_float(payload, ("limb_lengths_m", "tail"), value)
            return
        if field_id == "neck_length_m":
            self._set_nested_float(payload, ("limb_lengths_m", "neck"), value)
            return
        if field_id == "head_length_m":
            self._set_nested_float(payload, ("limb_lengths_m", "head"), value)
            payload["head_length_ratio"] = 1.0
            return
        if field_id == "thigh_length_m":
            self._set_nested_float(payload, ("limb_lengths_m", "thigh"), value)
            return
        if field_id == "shank_length_m":
            self._set_nested_float(payload, ("limb_lengths_m", "shank"), value)
            return
        if field_id == "metatarsus_length_m":
            self._set_nested_float(payload, ("limb_lengths_m", "metatarsus"), value)
            return
        if field_id == "foot_length_m":
            self._set_nested_float(payload, ("limb_lengths_m", "foot"), value)
            return

        if field_id == "rear_body_diameter_m":
            payload["rear_body_diameter_m"] = float(value)
            # keep the legacy ratio roughly aligned for downstream weighting / defaults
            current_front = float(payload.get("front_body_diameter_m", value * 0.85))
            payload["trunk_depth_ratio"] = (0.5 * (float(value) + current_front)) / h
            return
        if field_id == "front_body_diameter_m":
            payload["front_body_diameter_m"] = float(value)
            current_rear = float(payload.get("rear_body_diameter_m", value * 1.15))
            payload["trunk_depth_ratio"] = (0.5 * (current_rear + float(value))) / h
            return
        if field_id in {"tail_base_diameter_m", "tail_tip_diameter_m", "neck_diameter_m", "head_diameter_m", "thigh_diameter_m", "shank_diameter_m", "metatarsus_diameter_m"}:
            payload[field_id] = float(value)
            return

        if field_id == "rear_body_angle_deg":
            payload["rear_body_angle_deg"] = float(value)
            return
        if field_id == "front_body_angle_deg":
            payload["trunk_pitch_bias_deg"] = float(value)
            return
        if field_id == "tail_angle_deg":
            payload["tail_pitch_deg"] = float(value)
            return
        if field_id == "neck_angle_deg":
            payload["neck_angle_deg"] = float(value)
            # keep old fields coherent enough for legacy code paths
            payload["neck_min_pitch_deg"] = float(value) - 12.0
            payload["neck_max_pitch_deg"] = float(value) + 12.0
            return
        if field_id == "head_angle_deg":
            payload["head_angle_deg"] = float(value)
            payload["head_min_pitch_deg"] = float(value) - 10.0
            payload["head_max_pitch_deg"] = float(value) + 10.0
            return

    def _set_lab_editor_payload(self, payload: Dict[str, Any]) -> None:
        self.lab_payload = json.loads(json.dumps(payload))

    def lab_reload_controls_from_editor(self) -> None:
        try:
            if self.lab_payload is None:
                raise ValueError("No species selected.")
            self.lab_sync_controls_from_payload(self.lab_payload)
            self.lab_preview_status.configure(text="Controls reset to the current draft.", text_color=SUCCESS)
        except Exception as exc:
            messagebox.showerror("Reset controls failed", str(exc), parent=self)

    def lab_sync_controls_from_payload(self, payload: Dict[str, Any]) -> None:
        for spec in LAB_QUICK_FIELDS:
            value = self._control_value_from_payload(payload, spec["id"])
            if isinstance(value, (int, float)):
                text = f"{float(value):.4f}".rstrip("0").rstrip(".")
            else:
                text = str(value)
            var = self.lab_quick_vars.get(spec["id"])
            if var is not None:
                var.set(text)

    def lab_apply_quick_controls(self) -> None:
        try:
            payload = self.lab_current_payload()
            self._set_lab_editor_payload(payload)
            self._set_lab_draft(self.lab_group_var.get(), self.lab_selected_stem, payload)
            self.lab_update_preview(silent=True)
            self.lab_preview_status.configure(text="Preview updated. Draft ready for Generate.", text_color=SUCCESS)
            self.status_var.set("Morphology controls applied")
        except Exception as exc:
            messagebox.showerror("Apply controls failed", str(exc), parent=self)

    def lab_load_species(self, group: str, stem: str) -> None:
        path = get_species_path(group, stem)
        payload = self._get_lab_draft(group, stem)
        if payload is None:
            payload = json.loads(path.read_text(encoding="utf-8"))
        self._set_lab_editor_payload(payload)
        self.lab_sync_controls_from_payload(payload)
        suffix = " • draft" if self._get_lab_draft(group, stem) is not None else ""
        self.lab_path_label.configure(text=f"Species file: {path.name}{suffix}")
        self.lab_update_preview(silent=True)

    def lab_current_payload(self) -> Dict[str, Any]:
        base = json.loads(json.dumps(self.lab_payload or {}))
        if not base:
            raise ValueError("No species selected.")
        for spec in LAB_QUICK_FIELDS:
            raw = self.lab_quick_vars[spec["id"]].get().strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"{spec['label']} must be numeric.") from exc
            value = max(float(spec["min"]), min(float(spec["max"]), value))
            self._apply_control_to_payload(base, spec["id"], value)
        species_from_payload(base)
        return base

    def lab_update_preview(self, silent: bool = False) -> None:
        try:
            payload = self.lab_current_payload()
            self._set_lab_editor_payload(payload)
            render_species_preview(payload, self.lab_preview_path)
            self.lab_preview_player.load_image(self.lab_preview_path)
            summary = summarize_species_payload(payload)
            self.lab_summary.delete("1.0", "end")
            self.lab_summary.insert("1.0", "\n".join(f"{k}: {v}" for k, v in summary.items()))
            self.lab_preview_status.configure(text=f"Applied controls to preview: {payload.get('name', 'species')}", text_color=SUCCESS)
            self.status_var.set(f"Preview updated for {payload.get('name', 'species')}")
            if self.lab_live_viewer is not None and self.lab_live_viewer.winfo_exists():
                species = species_from_payload(payload)
                self.lab_live_viewer.set_sequence(build_species_pose(species), title=f"3D Morphology • {species.name}", kind="morphology")
        except Exception as exc:
            self.lab_preview_status.configure(text=f"Preview failed: {exc}", text_color=VIOLET_HOVER)
            if not silent:
                messagebox.showerror("Preview failed", str(exc), parent=self)
            self.lab_summary.delete("1.0", "end")
            self.lab_summary.insert("1.0", str(exc))

    def lab_save_current(self) -> None:
        try:
            payload = self.lab_current_payload()
            result = save_species_payload(payload, group=self.lab_group_var.get(), overwrite_stem=self.lab_selected_stem)
            self.lab_selected_stem = result["stem"]
            self._set_lab_editor_payload(payload)
            self._set_lab_draft(self.lab_group_var.get(), self.lab_selected_stem, payload)
            self.status_var.set(f"Saved {result['name']}")
            self.refresh_all()
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)

    def lab_save_as_new(self) -> None:
        try:
            payload = self.lab_current_payload()
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
            return
        dialog = ctk.CTkInputDialog(text="Enter a new file/name stem for this JSON:", title="Save as new")
        new_name = dialog.get_input() if dialog else None
        if not new_name:
            return
        try:
            old_stem = self.lab_selected_stem
            result = save_species_payload(payload, group=self.lab_group_var.get(), save_as_name=new_name)
            self.lab_selected_stem = result["stem"]
            self._set_lab_editor_payload(payload)
            self._clear_lab_draft(self.lab_group_var.get(), old_stem)
            self._set_lab_draft(self.lab_group_var.get(), self.lab_selected_stem, payload)
            self.status_var.set(f"Saved new species JSON: {result['stem']}")
            self.refresh_all()
        except Exception as exc:
            messagebox.showerror("Save as new failed", str(exc), parent=self)

    def _register_viewer(self, viewer: Interactive3DViewer) -> Interactive3DViewer:
        self.floating_viewers.append(viewer)
        def _cleanup() -> None:
            if viewer in self.floating_viewers:
                self.floating_viewers.remove(viewer)
            if viewer is self.lab_live_viewer:
                self.lab_live_viewer = None
            try:
                viewer.destroy()
            except Exception:
                pass
        viewer.protocol("WM_DELETE_WINDOW", _cleanup)
        return viewer

    def lab_open_3d_preview(self) -> None:
        try:
            self.lab_update_preview(silent=True)
            payload = self.lab_current_payload()
            species = species_from_payload(payload)
        except Exception as exc:
            messagebox.showerror("3D preview failed", str(exc), parent=self)
            return
        if self.lab_live_viewer is not None and self.lab_live_viewer.winfo_exists():
            self.lab_live_viewer.set_sequence(build_species_pose(species), title=f"3D Morphology • {species.name}", kind="morphology")
            self.lab_live_viewer.lift()
            self.lab_live_viewer.focus_force()
        else:
            viewer = Interactive3DViewer.from_species(self, species)
            self.lab_live_viewer = self._register_viewer(viewer)
        self.lab_preview_status.configure(text=f"Opened standard-pose 3D preview for {species.name}. Use Apply to keep it in sync.", text_color=SUCCESS)
        self.status_var.set(f"Opened standard-pose 3D preview for {species.name}")

    def _refresh_generate(self) -> None:
        target_values = [item["stem"] for item in self.targets] or [""]
        self.target_menu.configure(values=target_values)
        self.generate_target_info.configure(text=((f"{self.selected_target_details['name']} • {self.selected_target_details['height_m']:.2f} m • {self.selected_target_details['mass_kg']:.1f} kg" + (" • using unsaved draft" if self.selected_target_details.get("draft") else "")) if self.selected_target_details else "Select a target dinosaur."))
        self._clear(self.generate_clips_frame)
        self.generate_clip_vars = {}
        for clip in self.bird_clips:
            var = tk.BooleanVar(value=clip["clip_id"] in self.selected_clip_ids)
            self.generate_clip_vars[clip["clip_id"]] = var
            card = ctk.CTkFrame(self.generate_clips_frame, fg_color=CARD, corner_radius=18, border_width=1, border_color=BORDER)
            card.pack(fill="x", padx=6, pady=5)
            ctk.CTkCheckBox(card, text=clip["name"], variable=var, text_color=TEXT, command=self._sync_clip_selections_from_generate, fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=BORDER).pack(anchor="w", padx=12, pady=(10, 2))
            ctk.CTkLabel(card, text=f"{clip['species_name']} • {clip['csv_path']}", text_color=MUTED).pack(anchor="w", padx=14, pady=(0, 10))
        self.generate_clip_count.configure(text=f"{len(self.selected_clip_ids)} / {len(self.bird_clips)} clips selected")
        if self.latest_run_payload:
            self._populate_generate_report(self.latest_run_payload)

    def _refresh_history(self) -> None:
        self._clear(self.history_list)
        if not self.runs:
            ctk.CTkLabel(self.history_list, text="No runs recorded yet.", text_color=MUTED).pack(anchor="w", padx=10, pady=10)
            return
        for run in self.runs:
            self._run_card(self.history_list, run, compact=False).pack(fill="x", padx=6, pady=6)
        if self.runs:
            self.after(50, lambda payload=self.runs[0]: self._load_history_run(payload, switch_page=False))

    def _run_card(self, parent: ctk.CTkBaseClass, run: Dict[str, Any], compact: bool) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=20, border_width=1, border_color=BORDER)
        ctk.CTkLabel(card, text=f"{run.get('target_species', run.get('target_stem', 'target'))} • {run.get('run_name', '')}", font=ctk.CTkFont(size=18 if not compact else 16, weight="bold"), text_color=TEXT).pack(anchor="w", padx=14, pady=(12, 2))
        ctk.CTkLabel(card, text=f"Created: {run.get('created_at', '')}", text_color=MUTED).pack(anchor="w", padx=14)
        ctk.CTkButton(card, text="Inspect" if compact else "Open run", width=110, fg_color=INFO, hover_color="#4D7493", command=lambda payload=run: self._load_history_run(payload)).pack(anchor="w", padx=14, pady=(10, 12))
        return card

    def _select_target(self, stem: str) -> None:
        self.target_var.set(stem)
        self.selected_target_details = self._selected_target_summary()
        self._refresh_generate()

    def _sync_clip_selections_from_generate(self) -> None:
        self.selected_clip_ids = {cid for cid, var in self.generate_clip_vars.items() if var.get()}
        self.generate_clip_count.configure(text=f"{len(self.selected_clip_ids)} / {len(self.bird_clips)} clips selected")

    def select_all_clips(self) -> None:
        self.selected_clip_ids = {item["clip_id"] for item in self.bird_clips}
        self._refresh_generate()

    def clear_all_clips(self) -> None:
        self.selected_clip_ids = set()
        self._refresh_generate()

    def import_bird_species_flow(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            payload = import_species_json(path, group="bird")
            self.status_var.set(f"Imported bird species: {payload['name']}")
            self.lab_group_var.set("bird")
            self.lab_selected_stem = payload["stem"]
            self.refresh_all()
        except Exception as exc:
            messagebox.showerror("Bird JSON import failed", str(exc), parent=self)

    def import_target_species_flow(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            payload = import_species_json(path, group="theropod")
            self.status_var.set(f"Imported target species: {payload['name']}")
            self.target_var.set(payload["stem"])
            self.lab_group_var.set("theropod")
            self.lab_selected_stem = payload["stem"]
            self.refresh_all()
        except Exception as exc:
            messagebox.showerror("Target JSON import failed", str(exc), parent=self)

    def import_clip_flow(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        default_species = infer_bird_species_from_filename(path)
        dialog = SpeciesPickerDialog(self, self.bird_species, default_species=default_species)
        self.wait_window(dialog)
        if not dialog.result:
            return
        try:
            entry = import_bird_clip(path, species_stem=dialog.result["species_stem"], fps=float(dialog.result["fps"]), display_name=dialog.result["display_name"])
            self.status_var.set(f"Imported bird clip: {entry['name']}")
            self.refresh_all()
        except Exception as exc:
            messagebox.showerror("Bird CSV import failed", str(exc), parent=self)

    def preview_weights_flow(self) -> None:
        if not self.target_var.get():
            messagebox.showerror("No target selected", "Please choose a theropod target.", parent=self)
            return
        if not self.selected_clip_ids:
            messagebox.showerror("No clips selected", "Select at least one bird clip before previewing weights.", parent=self)
            return
        target_override, bird_overrides = self._collect_generation_overrides()
        self._run_async(
            "preview_weights",
            "Evaluating reference bird compatibility...",
            lambda: preview_weights(
                self.target_var.get(),
                clip_ids=sorted(self.selected_clip_ids),
                fps=float(self.fps_var.get()),
                target_payload_override=target_override,
                bird_species_payload_overrides=bird_overrides,
            ),
        )

    def generate_run_flow(self) -> None:
        if not self.target_var.get():
            messagebox.showerror("No target selected", "Please choose a theropod target.", parent=self)
            return
        if not self.selected_clip_ids:
            messagebox.showerror("No clips selected", "Select at least one bird clip before generating.", parent=self)
            return
        target_override, bird_overrides = self._collect_generation_overrides()
        self._run_async(
            "generate_run",
            "Generating fused theropod run...",
            lambda: generate_run(
                self.target_var.get(),
                clip_ids=sorted(self.selected_clip_ids),
                fps=float(self.fps_var.get()),
                repeat_cycles=int(self.repeat_var.get()),
                phase_frames=int(self.phase_var.get()),
                target_payload_override=target_override,
                bird_species_payload_overrides=bird_overrides,
            ),
        )

    def _run_async(self, kind: str, start_message: str, fn: Callable[[], Any]) -> None:
        self.status_var.set(start_message)
        self.progress.configure(mode="indeterminate")
        self.progress.start()

        def worker() -> None:
            try:
                self.worker_queue.put((kind, fn()))
            except Exception as exc:
                self.worker_queue.put((f"{kind}_error", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self.progress.set(1)
                if kind == "preview_weights":
                    self.status_var.set("Weight preview complete")
                    self._populate_weight_preview(payload)
                    self.show_page("generate")
                elif kind == "generate_run":
                    self.status_var.set("Run generation complete")
                    self.latest_run_payload = payload
                    self._populate_generate_report(payload)
                    self.refresh_all()
                    self.show_page("generate")
                elif kind.endswith("_error"):
                    self.status_var.set("Operation failed")
                    messagebox.showerror("Operation failed", str(payload), parent=self)
                self.progress.set(0)
        except queue.Empty:
            pass
        self.after(120, self._poll_worker_queue)

    def _populate_weight_preview(self, report: Any) -> None:
        self.generate_weight_chart.set_report(report)
        lines = [f"Target: {report.target_species}", f"Reference birds: {report.n_reference_birds}", "", "Automatic bird weighting:"]
        for row in report.contributions:
            lines.append(
                f"- {row['species']}: weight={float(row['weight']):.3f} | compat={float(row['compatibility']):.3f} | morph={float(row['morph_distance']):.3f} | posture={float(row['posture_distance']):.3f} | gait={float(row['gait_distance']):.3f}"
            )
        text = "\n".join(lines)
        self.generate_fusion_box.delete("1.0", "end")
        self.generate_fusion_box.insert("1.0", text)
        self.generate_session_box.delete("1.0", "end")
        self.generate_session_box.insert("1.0", text)
        self.generate_run_badge.configure(text="Preview only • no run folder written yet")

    def _populate_generate_report(self, payload: Dict[str, Any]) -> None:
        session = payload["session_report"]
        run_dir = Path(payload["run_dir"])
        gif_path = run_dir / session["outputs"]["dino_gif"]
        if gif_path.exists():
            self.generate_preview_player.load_gif(gif_path)
        self.generate_weight_chart.set_report(type("ReportView", (), {"contributions": session["fusion_report"]["contributions"], "target_species": session["target_species"], "n_reference_birds": len(session["fusion_report"]["contributions"])} )())
        self.generate_run_badge.configure(text=f"Latest run • {session['run_name']}")
        session_lines = [
            f"Run: {session['run_name']}",
            f"Target: {session['target_species']} ({session['target_stem']})",
            f"Created: {session['created_at']}",
            f"Output folder: {payload['run_dir']}",
            "",
            "Selected clips:",
        ]
        for clip in session["selected_clips"]:
            session_lines.append(f"- {clip['name']} ({clip['species_name']}, {clip['fps']:.1f} FPS)")
        session_lines.extend(["", "Reconstruction summary:", json.dumps(session.get("reconstruction_summary", {}), indent=2, ensure_ascii=False)])
        self.generate_session_box.delete("1.0", "end")
        self.generate_session_box.insert("1.0", "\n".join(session_lines))
        fusion_lines = ["Fusion weights:"]
        for row in session["fusion_report"]["contributions"]:
            fusion_lines.append(
                f"- {row['species']}: {float(row['weight']):.3f} | compat {float(row['compatibility']):.3f} | morph {float(row['morph_distance']):.3f} | posture {float(row['posture_distance']):.3f} | gait {float(row['gait_distance']):.3f}"
            )
        self.generate_fusion_box.delete("1.0", "end")
        self.generate_fusion_box.insert("1.0", "\n".join(fusion_lines))
        file_lines = [f"Run folder: {run_dir}", "", "Saved files:"] + [f"- {k}: {v}" for k, v in session["outputs"].items()]
        self.generate_files_box.delete("1.0", "end")
        self.generate_files_box.insert("1.0", "\n".join(file_lines))

    def _load_history_run(self, run_payload: Dict[str, Any], switch_page: bool = True) -> None:
        report_path = Path(run_payload["report_path"])
        self.last_history_report_path = report_path
        session = json.loads(report_path.read_text(encoding="utf-8"))
        gif_path = report_path.parent / session["outputs"]["dino_gif"]
        if gif_path.exists():
            self.history_preview_player.load_gif(gif_path)
        lines = [
            f"Run: {session['run_name']}",
            f"Target: {session['target_species']} ({session['target_stem']})",
            f"Created: {session['created_at']}",
            f"Report: {report_path}",
            "",
            "Selected clips:",
        ]
        for clip in session["selected_clips"]:
            lines.append(f"- {clip['name']} ({clip['species_name']}, {clip['fps']:.1f} FPS)")
        lines.extend(["", "Fusion weights:"])
        for row in session["fusion_report"]["contributions"]:
            lines.append(f"- {row['species']}: {float(row['weight']):.3f} (compat {float(row['compatibility']):.3f})")
        self.history_report_box.delete("1.0", "end")
        self.history_report_box.insert("1.0", "\n".join(lines))
        self.history_json_box.delete("1.0", "end")
        self.history_json_box.insert("1.0", json.dumps(session, indent=2, ensure_ascii=False))
        self.history_files_box.delete("1.0", "end")
        self.history_files_box.insert("1.0", "\n".join([f"Run folder: {report_path.parent}"] + [f"- {k}: {v}" for k, v in session["outputs"].items()]))
        if switch_page:
            self.show_page("history")

    def _export_file(self, src_path: str | Path, *, title: str, filetypes: list[tuple[str, str]]) -> None:
        src = Path(src_path)
        if not src.exists():
            messagebox.showerror("Missing file", f"Could not find {src}", parent=self)
            return
        dst = filedialog.asksaveasfilename(parent=self, title=title, initialfile=src.name, defaultextension=src.suffix, filetypes=filetypes)
        if not dst:
            return
        shutil.copy2(src, dst)
        self.status_var.set(f"Exported {src.name}")

    def export_latest_gif(self) -> None:
        if not self.latest_run_payload:
            messagebox.showinfo("No run yet", "Generate a run first.", parent=self)
            return
        session = self.latest_run_payload["session_report"]
        gif_path = Path(self.latest_run_payload["run_dir"]) / session["outputs"]["dino_gif"]
        self._export_file(gif_path, title="Export generated GIF", filetypes=[("GIF animation", "*.gif")])

    def export_history_gif(self) -> None:
        if not self.last_history_report_path:
            messagebox.showinfo("No run selected", "Choose a run from history first.", parent=self)
            return
        session = json.loads(self.last_history_report_path.read_text(encoding="utf-8"))
        gif_path = self.last_history_report_path.parent / session["outputs"]["dino_gif"]
        self._export_file(gif_path, title="Export saved GIF", filetypes=[("GIF animation", "*.gif")])

    def open_latest_run_3d_viewer(self) -> None:
        if not self.latest_run_payload:
            messagebox.showinfo("No run yet", "Generate a run first.", parent=self)
            return
        session = self.latest_run_payload["session_report"]
        csv_path = Path(self.latest_run_payload["run_dir"]) / session["outputs"]["dino_csv"]
        if not csv_path.exists():
            messagebox.showerror("Missing file", f"Could not find {csv_path}", parent=self)
            return
        viewer = Interactive3DViewer.from_csv(self, csv_path)
        self._register_viewer(viewer)

    def open_history_run_3d_viewer(self) -> None:
        if not self.last_history_report_path:
            messagebox.showinfo("No run selected", "Choose a run from history first.", parent=self)
            return
        session = json.loads(self.last_history_report_path.read_text(encoding="utf-8"))
        csv_path = self.last_history_report_path.parent / session["outputs"]["dino_csv"]
        if not csv_path.exists():
            messagebox.showerror("Missing file", f"Could not find {csv_path}", parent=self)
            return
        viewer = Interactive3DViewer.from_csv(self, csv_path)
        self._register_viewer(viewer)

    def show_page(self, name: str, instant: bool = False) -> None:
        if name == self.current_page:
            return
        old = self.page_frames.get(self.current_page)
        new = self.page_frames[name]
        title_map = {"dashboard": "Dashboard", "library": "Library", "lab": "Morphology Lab", "generate": "Generate", "history": "Run History"}
        self.page_title_var.set(title_map[name])
        for page, button in self.sidebar_buttons.items():
            button.configure(fg_color=VIOLET if page == name else "transparent")
        if instant or old is None:
            if old is not None:
                old.place_forget()
            new.place(in_=self.content, x=0, y=0, relwidth=1, relheight=1)
            self.current_page = name
            return
        new.place(in_=self.content, x=70, y=0, relwidth=1, relheight=1)

        def animate(step: int = 0) -> None:
            total = 10
            progress = step / total
            new.place_configure(x=int(70 * (1.0 - progress)))
            old.place_configure(x=int(-70 * progress))
            if step < total:
                self.after(18, lambda: animate(step + 1))
            else:
                old.place_forget()
                new.place_configure(x=0)

        animate()
        self.current_page = name

    def open_path(self, path: str | Path) -> None:
        try:
            webbrowser.open(Path(path).resolve().as_uri())
        except Exception:
            pass

    def capture_screenshot(self) -> None:
        if not self.screenshot_path:
            return
        self.update_idletasks()
        self.update()
        x, y, w, h = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        out_path = Path(self.screenshot_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

    @staticmethod
    def _clear(frame: ctk.CTkBaseClass) -> None:
        for child in frame.winfo_children():
            child.destroy()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", default="dashboard", choices=["dashboard", "library", "lab", "generate", "history"])
    parser.add_argument("--screenshot", default=None)
    parser.add_argument("--exit-after", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    app = JurassicGaitStudio(start_page=args.page, screenshot_path=args.screenshot, exit_after=args.exit_after)
    app.mainloop()


if __name__ == "__main__":
    main()
