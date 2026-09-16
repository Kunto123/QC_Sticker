from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable
import tkinter as tk

import customtkinter as ctk

import cv2
import numpy as np
from PIL import Image, ImageTk

from client_tk.app.theme import BORDER, PANEL_BG, TEXT_PRIMARY


_BBOX_COLOR = (34, 197, 94)
_POLYGON_COLOR = (250, 204, 21)
_DRAFT_COLOR = (248, 250, 252)
_TEXT_COLOR = (15, 23, 42)
_SELECTED_COLOR = (56, 189, 248)
_RESIZE_HANDLE_COLOR = (14, 116, 144)
_RESIZE_HANDLE_SIZE = 6
_RESIZE_HIT_RADIUS = 12


class AnnotationCanvas(ctk.CTkFrame):
    def __init__(self, master, title: str = "Image Annotation", *, size: tuple[int, int] | None = None):
        super().__init__(master, fg_color=PANEL_BG, corner_radius=14, border_width=1, border_color=BORDER)
        self._display_size = size or (960, 520)
        if size is not None:
            self.configure(width=size[0], height=size[1] + 34)
            self.pack_propagate(False)
            self.grid_propagate(False)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        canvas_width, canvas_height = self._display_size

        ctk.CTkLabel(self, text=title, font=("Segoe UI", 10, "bold"), text_color=TEXT_PRIMARY).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(10, 6),
        )

        self._canvas = tk.Canvas(
            self,
            width=canvas_width,
            height=canvas_height,
            bg="#0f172a",
            highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.grid(row=1, column=0, sticky="nw", padx=10, pady=(0, 10))
        self._canvas.bind("<Configure>", lambda _event: self.request_redraw())
        self._canvas.bind("<Map>", self._on_canvas_visible)
        self._canvas.bind("<Visibility>", self._on_canvas_visible)
        self._canvas.bind("<Expose>", self._on_canvas_visible)
        self._canvas.bind("<Enter>", self._on_canvas_visible)
        self._canvas.bind("<FocusIn>", self._on_canvas_visible)
        self._canvas.bind("<Button-1>", self._on_left_click)
        self._canvas.bind("<Button-3>", self._on_right_click)
        self._canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self._canvas.bind("<B3-Motion>", self._on_right_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self._canvas.bind("<ButtonRelease-3>", self._on_right_release)
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
        self._canvas.bind("<Escape>", self._cancel_draft)
        self._canvas.bind("<Delete>", self._on_delete_selected)
        self._canvas.focus_set()
        self.bind("<Map>", self._on_canvas_visible)
        self.bind("<Visibility>", self._on_canvas_visible)
        self.bind("<Expose>", self._on_canvas_visible)

        self._source_frame: np.ndarray | None = None
        self._image_name: str = ""
        self._class_name: str = "object"
        self._mode: str = "bbox"
        self._labels: list[dict] = []
        self._selected_label_index: int | None = None
        self._draft_bbox_start: tuple[int, int] | None = None
        self._draft_bbox_end: tuple[int, int] | None = None
        self._draft_polygon_points: list[tuple[int, int]] = []
        self._resize_label_index: int | None = None
        self._resize_handle: str | None = None
        self._resize_anchor_canvas: tuple[int, int] | None = None
        self._render_geometry: dict[str, int] | None = None
        self._pil_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._canvas_image_id: int | None = None
        self._redraw_job: str | None = None
        self._needs_redraw_on_map: bool = False
        self.on_labels_changed: Callable[[list[dict]], None] | None = None
        self.on_selection_changed: Callable[[dict | None, int | None], None] | None = None

    def set_mode(self, mode: str, *, redraw: bool = True) -> None:
        mode_name = str(mode or "bbox").strip().lower()
        if mode_name not in {"bbox", "polygon"}:
            mode_name = "bbox"
        if mode_name == self._mode:
            return
        self._mode = mode_name
        self._clear_draft()
        self._clear_resize_state()
        if redraw:
            self.redraw()

    def set_class_name(self, class_name: str, *, redraw: bool = True) -> None:
        self._class_name = str(class_name or "").strip() or "object"
        if self._source_frame is not None:
            if redraw:
                self.redraw()

    def set_image_name(self, image_name: str) -> None:
        self._image_name = str(image_name or "").strip()

    def load_bgr(self, frame: np.ndarray | None, *, image_name: str | None = None, redraw: bool = True) -> None:
        self._source_frame = frame.copy() if frame is not None else None
        if image_name is not None:
            self.set_image_name(image_name)
        self._clear_draft()
        if redraw:
            self.redraw()

    def load_image_path(self, image_path: str | Path, *, redraw: bool = True) -> bool:
        path = Path(image_path)
        frame = self._read_image(path)
        self._image_name = path.name
        self._source_frame = frame
        self._clear_draft()
        if redraw:
            self.redraw()
        return frame is not None

    def load_image_bytes(self, content: bytes, *, image_name: str | None = None, redraw: bool = True) -> bool:
        if image_name is not None:
            self.set_image_name(image_name)
        frame = self._read_image_bytes(content)
        self._source_frame = frame
        self._clear_draft()
        if redraw:
            self.redraw()
        return frame is not None

    def set_labels(self, labels: list[dict] | None, *, redraw: bool = True) -> None:
        self._labels = [copy.deepcopy(item) for item in labels or [] if isinstance(item, dict)]
        self._selected_label_index = None
        self._clear_draft()
        self._clear_resize_state()
        self._notify_selection_changed()
        if redraw:
            self.redraw()

    def get_labels(self) -> list[dict]:
        return [copy.deepcopy(item) for item in self._labels]

    def get_selected_label_index(self) -> int | None:
        if self._selected_label_index is None:
            return None
        if not (0 <= self._selected_label_index < len(self._labels)):
            return None
        return self._selected_label_index

    def set_selected_label_index(self, index: int | None) -> bool:
        if index is None:
            if self._selected_label_index is None:
                return False
            self._selected_label_index = None
            self._clear_resize_state()
            self._notify_selection_changed()
            self.request_redraw()
            return True
        if index < 0 or index >= len(self._labels):
            return False
        if self._selected_label_index == index:
            return True
        self._selected_label_index = index
        self._clear_resize_state()
        self._notify_selection_changed()
        self.request_redraw()
        return True

    def set_selected_label_class_name(self, class_name: str) -> bool:
        selected = self.get_selected_label_index()
        if selected is None:
            return False
        normalized = str(class_name or "").strip() or "object"
        label = self._labels[selected]
        label["class_name"] = normalized
        label["class"] = normalized
        self._class_name = normalized
        self.request_redraw()
        self._notify_selection_changed()
        self._notify_labels_changed()
        return True

    def delete_selected_label(self) -> bool:
        selected = self.get_selected_label_index()
        if selected is None:
            return False
        del self._labels[selected]
        self._selected_label_index = None
        self._clear_draft()
        self._clear_resize_state()
        self.request_redraw()
        self._notify_selection_changed()
        self._notify_labels_changed()
        return True

    def clear(self) -> None:
        self._source_frame = None
        self._labels = []
        self._selected_label_index = None
        self._clear_draft()
        self._clear_resize_state()
        self._render_geometry = None
        self._pil_image = None
        self._photo = None
        self._canvas_image_id = None
        self._canvas.delete("all")
        self._canvas.configure(bg="#0f172a")
        self._notify_selection_changed()

    def request_redraw(self) -> None:
        if self._redraw_job is not None:
            return
        self._redraw_job = self.after_idle(self._run_scheduled_redraw)

    def _run_scheduled_redraw(self) -> None:
        self._redraw_job = None
        self.redraw()

    def redraw(self) -> None:
        if not self.winfo_ismapped() or not self._canvas.winfo_ismapped():
            self._needs_redraw_on_map = True
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw <= 8:
            cw = self._display_size[0]
        if ch <= 8:
            ch = self._display_size[1]

        if self._source_frame is None:
            self._render_geometry = None
            self._photo = None
            self._canvas_image_id = None
            self._canvas.delete("all")
            self._canvas.create_text(
                cw // 2,
                ch // 2,
                text="Belum ada gambar.\nPilih dataset lalu gunakan tombol Previous / Next.",
                fill="#64748b",
                font=("Segoe UI", 11),
                justify="center",
            )
            self._needs_redraw_on_map = False
            return

        canvas_frame = self._build_canvas_frame(cw, ch)
        self._draw_labels(canvas_frame)
        self._draw_draft(canvas_frame)

        rgb = cv2.cvtColor(canvas_frame, cv2.COLOR_BGR2RGB)
        self._pil_image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(self._pil_image)
        self._canvas.delete("all")
        self._canvas_image_id = self._canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._needs_redraw_on_map = False

    def _read_image(self, path: Path) -> np.ndarray | None:
        image = cv2.imread(str(path))
        if image is not None:
            return image
        try:
            raw = np.fromfile(str(path), dtype=np.uint8)
        except Exception:  # noqa: BLE001
            return None
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)

    def _read_image_bytes(self, content: bytes | bytearray | memoryview | None) -> np.ndarray | None:
        if not content:
            return None
        raw = np.frombuffer(bytes(content), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)

    def _build_canvas_frame(self, cw: int, ch: int) -> np.ndarray:
        assert self._source_frame is not None
        source = self._source_frame
        src_h, src_w = source.shape[:2]
        scale = min(cw / max(src_w, 1), ch / max(src_h, 1))
        display_w = max(1, int(src_w * scale))
        display_h = max(1, int(src_h * scale))
        resized = cv2.resize(source, (display_w, display_h), interpolation=cv2.INTER_AREA)
        canvas_frame = np.zeros((ch, cw, 3), dtype=np.uint8)
        offset_x = (cw - display_w) // 2
        offset_y = (ch - display_h) // 2
        canvas_frame[offset_y:offset_y + display_h, offset_x:offset_x + display_w] = resized
        self._render_geometry = {
            "canvas_width": cw,
            "canvas_height": ch,
            "display_width": display_w,
            "display_height": display_h,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "source_width": src_w,
            "source_height": src_h,
        }
        return canvas_frame

    def _draw_labels(self, frame: np.ndarray) -> None:
        for index, label in enumerate(self._labels):
            selected = index == self._selected_label_index
            shape = self._label_shape(label)
            if shape == "polygon":
                self._draw_polygon_label(frame, label, selected=selected)
            else:
                self._draw_bbox_label(frame, label, selected=selected)

    def _draw_draft(self, frame: np.ndarray) -> None:
        if self._mode == "bbox" and self._draft_bbox_start and self._draft_bbox_end:
            x1, y1 = self._draft_bbox_start
            x2, y2 = self._draft_bbox_end
            cv2.rectangle(frame, (x1, y1), (x2, y2), _DRAFT_COLOR, 2)
            self._draw_text(frame, f"draft: {self._class_name}", x1, max(18, y1 - 8), _DRAFT_COLOR)
        elif self._mode == "polygon" and self._draft_polygon_points:
            points = np.array(self._draft_polygon_points, dtype=np.int32)
            cv2.polylines(frame, [points], False, _DRAFT_COLOR, 2, cv2.LINE_AA)
            for index, (px, py) in enumerate(self._draft_polygon_points):
                cv2.circle(frame, (px, py), 4, _DRAFT_COLOR, 1, cv2.LINE_AA)
                if index == 0:
                    self._draw_text(frame, f"{self._class_name} (polygon)", px + 8, max(18, py - 8), _DRAFT_COLOR)

    def _draw_bbox_label(self, frame: np.ndarray, label: dict, *, selected: bool = False) -> None:
        box = None
        if isinstance(label.get("bbox"), dict):
            box = label["bbox"]
        elif all(key in label for key in ("x", "y", "w", "h")):
            box = label
        if not isinstance(box, dict):
            return
        geometry = self._render_geometry
        if geometry is None:
            return
        values = self._normalize_box_values(box, bool(label.get("normalized", True)), geometry)
        if values is None:
            return
        x1, y1, x2, y2 = values
        class_name = self._label_class_name(label)
        color = _SELECTED_COLOR if selected else _BBOX_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if selected else 2)
        label_text = f"selected: {class_name}" if selected else (class_name or self._class_name)
        self._draw_text(frame, label_text, x1, max(18, y1 - 8), color)
        if selected:
            self._draw_bbox_handles(frame, x1, y1, x2, y2)

    def _draw_bbox_handles(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        handles = self._bbox_handles_from_canvas_box(x1, y1, x2, y2)
        half = _RESIZE_HANDLE_SIZE
        for center_x, center_y in handles.values():
            left = max(0, center_x - half)
            top = max(0, center_y - half)
            right = min(frame.shape[1] - 1, center_x + half)
            bottom = min(frame.shape[0] - 1, center_y + half)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), -1)
            cv2.rectangle(frame, (left, top), (right, bottom), _RESIZE_HANDLE_COLOR, 1)

    def _draw_polygon_label(self, frame: np.ndarray, label: dict, *, selected: bool = False) -> None:
        raw_points = label.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < 3:
            return
        geometry = self._render_geometry
        if geometry is None:
            return
        points = self._normalize_polygon_points(raw_points, bool(label.get("normalized", True)), geometry)
        if len(points) < 3:
            return
        poly = np.array(points, dtype=np.int32)
        color = _SELECTED_COLOR if selected else _POLYGON_COLOR
        cv2.polylines(frame, [poly], True, color, 3 if selected else 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, color, 1, cv2.LINE_AA)
        class_name = self._label_class_name(label)
        first_x, first_y = points[0]
        label_text = f"selected: {class_name}" if selected else (class_name or self._class_name)
        self._draw_text(frame, label_text, first_x, max(18, first_y - 8), color)

    def _draw_text(self, frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        if not text:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad_x = 5
        pad_y = 4
        left = max(0, x)
        top = max(0, y - text_height - baseline - pad_y)
        right = min(frame.shape[1] - 1, left + text_width + pad_x * 2)
        bottom = min(frame.shape[0] - 1, top + text_height + baseline + pad_y * 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), -1)
        cv2.putText(
            frame,
            text,
            (left + pad_x, bottom - pad_y - baseline),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _normalize_box_values(self, box: dict, normalized: bool, geometry: dict[str, int]) -> tuple[int, int, int, int] | None:
        try:
            x = float(box.get("x"))
            y = float(box.get("y"))
            width = float(box.get("w"))
            height = float(box.get("h"))
        except (TypeError, ValueError):
            return None
        display_w = geometry["display_width"]
        display_h = geometry["display_height"]
        offset_x = geometry["offset_x"]
        offset_y = geometry["offset_y"]
        source_w = geometry["source_width"]
        source_h = geometry["source_height"]

        if normalized:
            start_x = offset_x + int(x * display_w)
            start_y = offset_y + int(y * display_h)
            end_x = offset_x + int((x + width) * display_w)
            end_y = offset_y + int((y + height) * display_h)
        else:
            start_x = offset_x + int((x / max(source_w, 1)) * display_w)
            start_y = offset_y + int((y / max(source_h, 1)) * display_h)
            end_x = offset_x + int(((x + width) / max(source_w, 1)) * display_w)
            end_y = offset_y + int(((y + height) / max(source_h, 1)) * display_h)

        x1, x2 = sorted((start_x, end_x))
        y1, y2 = sorted((start_y, end_y))
        return x1, y1, x2, y2

    def _normalize_polygon_points(
        self,
        raw_points: list[dict],
        normalized: bool,
        geometry: dict[str, int],
    ) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        display_w = geometry["display_width"]
        display_h = geometry["display_height"]
        offset_x = geometry["offset_x"]
        offset_y = geometry["offset_y"]
        source_w = geometry["source_width"]
        source_h = geometry["source_height"]

        for point in raw_points:
            if not isinstance(point, dict):
                continue
            try:
                px = float(point.get("x"))
                py = float(point.get("y"))
            except (TypeError, ValueError):
                continue
            if normalized:
                canvas_x = offset_x + int(px * display_w)
                canvas_y = offset_y + int(py * display_h)
            else:
                canvas_x = offset_x + int((px / max(source_w, 1)) * display_w)
                canvas_y = offset_y + int((py / max(source_h, 1)) * display_h)
            points.append((canvas_x, canvas_y))
        return points

    def _label_shape(self, label: dict) -> str:
        value = str(label.get("type") or label.get("shape_type") or label.get("kind") or "bbox").strip().lower()
        return value or "bbox"

    def _label_class_name(self, label: dict) -> str:
        return str(label.get("class_name") or label.get("class") or label.get("label") or self._class_name or "object").strip() or self._class_name

    def _canvas_to_normalized(self, x: int, y: int) -> tuple[float, float] | None:
        geometry = self._render_geometry
        if geometry is None:
            return None
        offset_x = geometry["offset_x"]
        offset_y = geometry["offset_y"]
        display_w = geometry["display_width"]
        display_h = geometry["display_height"]
        if x < offset_x or y < offset_y or x > offset_x + display_w or y > offset_y + display_h:
            return None
        norm_x = max(0.0, min(1.0, (x - offset_x) / max(display_w, 1)))
        norm_y = max(0.0, min(1.0, (y - offset_y) / max(display_h, 1)))
        return norm_x, norm_y

    def _finish_bbox(self) -> None:
        if self._draft_bbox_start is None or self._draft_bbox_end is None:
            return
        geometry = self._render_geometry
        if geometry is None:
            return
        start = self._canvas_to_normalized(*self._draft_bbox_start)
        end = self._canvas_to_normalized(*self._draft_bbox_end)
        if start is None or end is None:
            self._clear_draft()
            self.redraw()
            return
        x1, y1 = start
        x2, y2 = end
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        if width < 0.005 or height < 0.005:
            self._clear_draft()
            self.redraw()
            return
        self._labels.append(
            {
                "type": "bbox",
                "shape_type": "bbox",
                "class_name": self._class_name,
                "class": self._class_name,
                "bbox": {
                    "x": round(left, 6),
                    "y": round(top, 6),
                    "w": round(width, 6),
                    "h": round(height, 6),
                },
                "normalized": True,
                "source": "visual",
            }
        )
        self._selected_label_index = len(self._labels) - 1
        self._clear_draft()
        self.request_redraw()
        self._notify_selection_changed()
        self._notify_labels_changed()

    def _finish_polygon(self) -> None:
        if len(self._draft_polygon_points) < 3:
            return
        geometry = self._render_geometry
        if geometry is None:
            return
        points: list[dict[str, float]] = []
        for canvas_x, canvas_y in self._draft_polygon_points:
            normalized = self._canvas_to_normalized(canvas_x, canvas_y)
            if normalized is None:
                continue
            points.append({"x": round(normalized[0], 6), "y": round(normalized[1], 6)})
        if len(points) < 3:
            self._clear_draft()
            self.redraw()
            return
        self._labels.append(
            {
                "type": "polygon",
                "shape_type": "polygon",
                "class_name": self._class_name,
                "class": self._class_name,
                "points": points,
                "normalized": True,
                "source": "visual",
            }
        )
        self._selected_label_index = len(self._labels) - 1
        self._clear_draft()
        self.request_redraw()
        self._notify_selection_changed()
        self._notify_labels_changed()

    def _notify_labels_changed(self) -> None:
        if self.on_labels_changed is None:
            return
        self.on_labels_changed(self.get_labels())

    def _notify_selection_changed(self) -> None:
        if self.on_selection_changed is None:
            return
        index = self.get_selected_label_index()
        if index is None:
            self.on_selection_changed(None, None)
            return
        self.on_selection_changed(copy.deepcopy(self._labels[index]), index)

    def _clear_draft(self) -> None:
        self._draft_bbox_start = None
        self._draft_bbox_end = None
        self._draft_polygon_points = []

    def _clear_resize_state(self) -> None:
        self._resize_label_index = None
        self._resize_handle = None
        self._resize_anchor_canvas = None

    def _cancel_draft(self, _event=None) -> None:
        self._clear_draft()
        self._clear_resize_state()
        self.request_redraw()

    def _on_left_click(self, event: tk.Event) -> None:
        if self._source_frame is None:
            return
        self._canvas.focus_set()
        if self._resize_label_index is not None:
            return
        if self._mode == "polygon":
            if self._draft_polygon_points and len(self._draft_polygon_points) >= 3:
                last_point = self._draft_polygon_points[-1]
                if abs(last_point[0] - event.x) <= 4 and abs(last_point[1] - event.y) <= 4:
                    self._finish_polygon()
                    return
            if self._selected_label_index is not None:
                self.set_selected_label_index(None)
            self._draft_polygon_points.append((int(event.x), int(event.y)))
            self.request_redraw()
            return
        if self._selected_label_index is not None:
            self.set_selected_label_index(None)
        self._draft_bbox_start = (int(event.x), int(event.y))
        self._draft_bbox_end = (int(event.x), int(event.y))
        self.request_redraw()

    def _on_right_click(self, event: tk.Event) -> None:
        if self._source_frame is None:
            return
        self._canvas.focus_set()
        if self._draft_bbox_start is not None or self._draft_polygon_points:
            return
        if self._try_begin_bbox_resize(int(event.x), int(event.y)):
            return
        selected = self._hit_test_label_index(int(event.x), int(event.y))
        self.set_selected_label_index(selected)

    def _on_right_drag(self, event: tk.Event) -> None:
        if self._source_frame is None:
            return
        if self._resize_label_index is None or self._resize_anchor_canvas is None:
            return
        self._resize_selected_bbox_to_canvas(int(event.x), int(event.y))

    def _on_right_release(self, _event: tk.Event) -> None:
        if self._resize_label_index is None:
            return
        self._clear_resize_state()
        self.request_redraw()
        self._notify_labels_changed()

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self._source_frame is None:
            return
        if self._mode == "bbox" and self._draft_bbox_start is not None:
            self._draft_bbox_end = (int(event.x), int(event.y))
            self.request_redraw()

    def _on_left_release(self, event: tk.Event) -> None:
        if self._source_frame is None:
            return
        if self._mode == "bbox" and self._draft_bbox_start is not None:
            self._draft_bbox_end = (int(event.x), int(event.y))
            self._finish_bbox()

    def _on_double_click(self, event: tk.Event) -> None:
        if self._source_frame is None:
            return
        if self._mode != "polygon":
            return
        if not self._draft_polygon_points:
            return
        if len(self._draft_polygon_points) >= 2:
            last_point = self._draft_polygon_points[-1]
            previous_point = self._draft_polygon_points[-2]
            if abs(last_point[0] - previous_point[0]) <= 4 and abs(last_point[1] - previous_point[1]) <= 4:
                self._draft_polygon_points.pop()
        self._finish_polygon()

    def _on_mouse_leave(self, _event=None) -> None:
        if self._mode == "bbox" and self._draft_bbox_start is not None and self._draft_bbox_end is not None:
            self.request_redraw()

    def _on_canvas_visible(self, _event=None) -> None:
        if self._source_frame is None:
            return
        if self._needs_redraw_on_map or self._photo is None:
            self.request_redraw()

    def _on_delete_selected(self, _event=None) -> None:
        self.delete_selected_label()

    def _try_begin_bbox_resize(self, canvas_x: int, canvas_y: int) -> bool:
        selected = self.get_selected_label_index()
        if selected is None:
            return False
        if selected >= len(self._labels):
            return False
        label = self._labels[selected]
        if self._label_shape(label) != "bbox":
            return False
        canvas_box = self._label_bbox_canvas_box(label)
        if canvas_box is None:
            return False
        x1, y1, x2, y2 = canvas_box
        handles = self._bbox_handles_from_canvas_box(x1, y1, x2, y2)
        nearest = self._nearest_handle(handles, canvas_x, canvas_y)
        if nearest is None:
            return False
        handle_name, _distance = nearest
        anchor_lookup = {
            "nw": (x2, y2),
            "ne": (x1, y2),
            "sw": (x2, y1),
            "se": (x1, y1),
        }
        self._resize_label_index = selected
        self._resize_handle = handle_name
        self._resize_anchor_canvas = anchor_lookup[handle_name]
        self.request_redraw()
        return True

    def _resize_selected_bbox_to_canvas(self, canvas_x: int, canvas_y: int) -> None:
        if self._resize_label_index is None or self._resize_anchor_canvas is None:
            return
        if self._resize_label_index >= len(self._labels):
            self._clear_resize_state()
            return
        label = self._labels[self._resize_label_index]
        clamped_x, clamped_y = self._clamp_canvas_point_to_image(canvas_x, canvas_y)
        anchor_x, anchor_y = self._resize_anchor_canvas
        anchor_norm = self._canvas_to_normalized(anchor_x, anchor_y)
        current_norm = self._canvas_to_normalized(clamped_x, clamped_y)
        if anchor_norm is None or current_norm is None:
            return

        left = min(anchor_norm[0], current_norm[0])
        top = min(anchor_norm[1], current_norm[1])
        width = abs(current_norm[0] - anchor_norm[0])
        height = abs(current_norm[1] - anchor_norm[1])

        min_size = 0.003
        if width < min_size:
            width = min_size
            if current_norm[0] < anchor_norm[0]:
                left = max(0.0, anchor_norm[0] - width)
            else:
                left = min(anchor_norm[0], 1.0 - width)
        if height < min_size:
            height = min_size
            if current_norm[1] < anchor_norm[1]:
                top = max(0.0, anchor_norm[1] - height)
            else:
                top = min(anchor_norm[1], 1.0 - height)

        left = max(0.0, min(left, 1.0 - width))
        top = max(0.0, min(top, 1.0 - height))
        width = max(min_size, min(width, 1.0 - left))
        height = max(min_size, min(height, 1.0 - top))

        box_payload = {
            "x": round(left, 6),
            "y": round(top, 6),
            "w": round(width, 6),
            "h": round(height, 6),
        }
        if isinstance(label.get("bbox"), dict):
            label["bbox"] = box_payload
        else:
            label.update(box_payload)
        label["normalized"] = True
        self.request_redraw()

    def _clamp_canvas_point_to_image(self, canvas_x: int, canvas_y: int) -> tuple[int, int]:
        geometry = self._render_geometry
        if geometry is None:
            return canvas_x, canvas_y
        min_x = geometry["offset_x"]
        min_y = geometry["offset_y"]
        max_x = geometry["offset_x"] + geometry["display_width"]
        max_y = geometry["offset_y"] + geometry["display_height"]
        clamped_x = max(min_x, min(canvas_x, max_x))
        clamped_y = max(min_y, min(canvas_y, max_y))
        return int(clamped_x), int(clamped_y)

    def _label_bbox_canvas_box(self, label: dict) -> tuple[int, int, int, int] | None:
        geometry = self._render_geometry
        if geometry is None:
            return None
        box_source = None
        if isinstance(label.get("bbox"), dict):
            box_source = label.get("bbox")
        elif all(key in label for key in ("x", "y", "w", "h")):
            box_source = label
        if not isinstance(box_source, dict):
            return None
        return self._normalize_box_values(box_source, bool(label.get("normalized", True)), geometry)

    def _bbox_handles_from_canvas_box(self, x1: int, y1: int, x2: int, y2: int) -> dict[str, tuple[int, int]]:
        return {
            "nw": (x1, y1),
            "ne": (x2, y1),
            "sw": (x1, y2),
            "se": (x2, y2),
        }

    def _nearest_handle(
        self,
        handles: dict[str, tuple[int, int]],
        canvas_x: int,
        canvas_y: int,
    ) -> tuple[str, float] | None:
        nearest_name = ""
        nearest_distance = float("inf")
        for handle_name, (handle_x, handle_y) in handles.items():
            distance = float(np.hypot(canvas_x - handle_x, canvas_y - handle_y))
            if distance < nearest_distance:
                nearest_name = handle_name
                nearest_distance = distance
        if nearest_distance > _RESIZE_HIT_RADIUS:
            return None
        return nearest_name, nearest_distance

    def _hit_test_label_index(self, canvas_x: int, canvas_y: int) -> int | None:
        normalized = self._canvas_to_normalized(canvas_x, canvas_y)
        if normalized is None:
            return None
        point_x, point_y = normalized
        for index in range(len(self._labels) - 1, -1, -1):
            label = self._labels[index]
            if self._label_contains_point(label, point_x, point_y):
                return index
        return None

    def _label_contains_point(self, label: dict, point_x: float, point_y: float) -> bool:
        shape = self._label_shape(label)
        if shape == "polygon":
            polygon = self._label_polygon_normalized(label)
            if len(polygon) < 3:
                return False
            return self._point_in_polygon(point_x, point_y, polygon)
        box = self._label_box_normalized(label)
        if box is None:
            return False
        left, top, right, bottom = box
        return left <= point_x <= right and top <= point_y <= bottom

    def _label_box_normalized(self, label: dict) -> tuple[float, float, float, float] | None:
        box = None
        if isinstance(label.get("bbox"), dict):
            box = label["bbox"]
        elif all(key in label for key in ("x", "y", "w", "h")):
            box = label
        if not isinstance(box, dict):
            return None
        values = self._load_box_values(box)
        if values is None:
            return None
        x, y, width, height = values
        if not bool(label.get("normalized", True)):
            source_h, source_w = self._source_dimensions()
            if source_w <= 0 or source_h <= 0:
                return None
            x /= float(source_w)
            y /= float(source_h)
            width /= float(source_w)
            height /= float(source_h)
        left = min(x, x + width)
        right = max(x, x + width)
        top = min(y, y + height)
        bottom = max(y, y + height)
        return left, top, right, bottom

    def _label_polygon_normalized(self, label: dict) -> list[tuple[float, float]]:
        raw_points = label.get("points")
        if not isinstance(raw_points, list):
            return []
        normalized = bool(label.get("normalized", True))
        source_h, source_w = self._source_dimensions()
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            try:
                px = float(point.get("x"))
                py = float(point.get("y"))
            except (TypeError, ValueError):
                continue
            if not normalized:
                if source_w <= 0 or source_h <= 0:
                    continue
                px /= float(source_w)
                py /= float(source_h)
            points.append((px, py))
        return points

    def _point_in_polygon(self, point_x: float, point_y: float, polygon: list[tuple[float, float]]) -> bool:
        inside = False
        count = len(polygon)
        j = count - 1
        for i in range(count):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = (yi > point_y) != (yj > point_y)
            if intersects:
                denominator = (yj - yi)
                if abs(denominator) < 1e-9:
                    j = i
                    continue
                cross_x = ((xj - xi) * (point_y - yi) / denominator) + xi
                if point_x <= cross_x:
                    inside = not inside
            j = i
        return inside

    def _load_box_values(self, box: dict) -> tuple[float, float, float, float] | None:
        try:
            x = float(box.get("x"))
            y = float(box.get("y"))
            width = float(box.get("w"))
            height = float(box.get("h"))
        except (TypeError, ValueError):
            return None
        return x, y, width, height

    def _source_dimensions(self) -> tuple[int, int]:
        if self._source_frame is None:
            return 0, 0
        height, width = self._source_frame.shape[:2]
        return int(height), int(width)
