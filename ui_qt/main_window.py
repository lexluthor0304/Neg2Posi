from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class ProcessingAPI:
    preview_images: Callable[[str, str], Tuple[np.ndarray, np.ndarray]]
    process_path: Callable[[str, str, str, bool, Optional[MutableMapping[str, Iterable[Tuple[float, float]]]]], None]
    apply_crop_editor: Callable[[str, Dict[str, float], float, str], Tuple[np.ndarray, np.ndarray]]
    get_crop_settings: Callable[[str], Tuple[Dict[str, float], float]]
    user_crops: MutableMapping[str, Iterable[Tuple[float, float]]]
    user_angles: MutableMapping[str, float]
    allowed_extensions: Tuple[str, ...]


I18N_DIR = Path(__file__).resolve().parents[1] / "resources" / "i18n"


def _load_language_file(code: str) -> Dict[str, str]:
    path = I18N_DIR / f"{code}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def numpy_to_qpixmap(arr: np.ndarray) -> QtGui.QPixmap:
    if arr is None:
        return QtGui.QPixmap()
    if arr.ndim == 2:
        normalized = np.clip(arr, 0.0, 1.0) if arr.dtype.kind == "f" else arr
        data = (normalized * 255).astype(np.uint8) if normalized.dtype != np.uint8 else normalized
        h, w = data.shape
        qimg = QtGui.QImage(data.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        return QtGui.QPixmap.fromImage(qimg.copy())
    if arr.ndim == 3:
        if arr.dtype.kind == "f":
            data = np.clip(arr, 0.0, 1.0)
            data = (data * 255).astype(np.uint8)
        else:
            data = arr.astype(np.uint8)
        if data.shape[2] == 3:
            h, w, _ = data.shape
            bytes_per_line = 3 * w
            qimg = QtGui.QImage(data.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            return QtGui.QPixmap.fromImage(qimg.copy())
        if data.shape[2] == 4:
            h, w, _ = data.shape
            bytes_per_line = 4 * w
            qimg = QtGui.QImage(data.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
            return QtGui.QPixmap.fromImage(qimg.copy())
    raise ValueError("Unsupported image data for preview")


class CropDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        translations: Dict[str, str],
        margins: Dict[str, float],
        angle: float,
        img_shape: Tuple[int, int],
    ) -> None:
        super().__init__(parent)
        self._t = translations
        self.setWindowTitle(self._t.get("crop_editor_title", "Edit Crop"))
        self.setModal(True)
        h, w = img_shape

        layout = QtWidgets.QFormLayout(self)

        self.left_spin = QtWidgets.QSpinBox()
        self.left_spin.setRange(0, max(0, w - 1))
        self.left_spin.setValue(int(round(margins.get("left", 0.0))))

        self.right_spin = QtWidgets.QSpinBox()
        self.right_spin.setRange(0, max(0, w - 1))
        self.right_spin.setValue(int(round(margins.get("right", 0.0))))

        self.top_spin = QtWidgets.QSpinBox()
        self.top_spin.setRange(0, max(0, h - 1))
        self.top_spin.setValue(int(round(margins.get("top", 0.0))))

        self.bottom_spin = QtWidgets.QSpinBox()
        self.bottom_spin.setRange(0, max(0, h - 1))
        self.bottom_spin.setValue(int(round(margins.get("bottom", 0.0))))

        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(-180.0, 180.0)
        self.angle_spin.setDecimals(2)
        self.angle_spin.setSingleStep(0.5)
        self.angle_spin.setValue(float(angle))

        layout.addRow(self._t.get("crop_margin_left", "Left (px)"), self.left_spin)
        layout.addRow(self._t.get("crop_margin_right", "Right (px)"), self.right_spin)
        layout.addRow(self._t.get("crop_margin_top", "Top (px)"), self.top_spin)
        layout.addRow(self._t.get("crop_margin_bottom", "Bottom (px)"), self.bottom_spin)
        layout.addRow(self._t.get("crop_rotation_angle", "Rotation Angle (°)"), self.angle_spin)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.button(QtWidgets.QDialogButtonBox.Ok).setText(self._t.get("dialog_crop_apply", "Apply"))
        button_box.button(QtWidgets.QDialogButtonBox.Cancel).setText(self._t.get("dialog_crop_cancel", "Cancel"))
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def values(self) -> Tuple[Dict[str, float], float]:
        return {
            "left": float(self.left_spin.value()),
            "right": float(self.right_spin.value()),
            "top": float(self.top_spin.value()),
            "bottom": float(self.bottom_spin.value()),
        }, float(self.angle_spin.value())


class CropPreview(QtWidgets.QFrame):
    cropChanged = QtCore.Signal(QtCore.QRectF)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setLineWidth(1)
        self.setMinimumSize(200, 200)
        self.setCursor(QtCore.Qt.CrossCursor)
        self._pixmap = QtGui.QPixmap()
        self._scaled_pixmap = QtGui.QPixmap()
        self._image_size = QtCore.QSize()
        self._offset = QtCore.QPointF(0.0, 0.0)
        self._scale = 1.0
        self._crop_rect = QtCore.QRectF()
        self._dragging = False
        self._drag_start = QtCore.QPointF()
        self._drag_current = QtCore.QPointF()
        self._drag_rect_widget = QtCore.QRectF()

    def sizeHint(self) -> QtCore.QSize:  # noqa: D401, N802
        return QtCore.QSize(320, 320)

    def clear(self) -> None:
        self._pixmap = QtGui.QPixmap()
        self._scaled_pixmap = QtGui.QPixmap()
        self._image_size = QtCore.QSize()
        self._crop_rect = QtCore.QRectF()
        self.update()

    def set_image(
        self,
        arr: Optional[np.ndarray],
        crop_rect: Optional[QtCore.QRectF] = None,
    ) -> None:
        if arr is None:
            self.clear()
            return
        self._pixmap = numpy_to_qpixmap(arr)
        self._image_size = QtCore.QSize(self._pixmap.width(), self._pixmap.height())
        self._update_scaled_pixmap()
        if crop_rect is None or crop_rect.isNull():
            self._crop_rect = self._full_image_rect()
        else:
            self._crop_rect = self._clip_rect_to_image(crop_rect)
        self.update()

    def set_crop_rect(self, crop_rect: Optional[QtCore.QRectF]) -> None:
        if self._image_size.isEmpty():
            return
        if crop_rect is None or crop_rect.isNull():
            self._crop_rect = self._full_image_rect()
        else:
            self._crop_rect = self._clip_rect_to_image(crop_rect)
        self.update()

    def _full_image_rect(self) -> QtCore.QRectF:
        if self._image_size.isEmpty():
            return QtCore.QRectF()
        return QtCore.QRectF(
            0.0,
            0.0,
            float(self._image_size.width()),
            float(self._image_size.height()),
        )

    def _clip_rect_to_image(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        if self._image_size.isEmpty():
            return QtCore.QRectF()
        rect = rect.normalized()
        width = float(self._image_size.width())
        height = float(self._image_size.height())
        x0 = min(max(rect.x(), 0.0), width)
        y0 = min(max(rect.y(), 0.0), height)
        x1 = min(max(rect.x() + rect.width(), 0.0), width)
        y1 = min(max(rect.y() + rect.height(), 0.0), height)
        if x1 - x0 <= 0.0 or y1 - y0 <= 0.0:
            return self._full_image_rect()
        return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap.isNull():
            self._scaled_pixmap = QtGui.QPixmap()
            self._scale = 1.0
            self._offset = QtCore.QPointF(0.0, 0.0)
            return
        self._scaled_pixmap = self._pixmap.scaled(
            self.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        if self._pixmap.width() == 0:
            self._scale = 1.0
        else:
            self._scale = self._scaled_pixmap.width() / self._pixmap.width()
        self._offset = QtCore.QPointF(
            (self.width() - self._scaled_pixmap.width()) / 2.0,
            (self.height() - self._scaled_pixmap.height()) / 2.0,
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if not self._scaled_pixmap.isNull():
            target = QtCore.QPointF(self._offset)
            painter.drawPixmap(target, self._scaled_pixmap)
            if not self._crop_rect.isNull():
                rect_widget = self._image_rect_to_widget(self._crop_rect)
                overlay_color = QtGui.QColor(0, 180, 255)
                painter.setPen(QtGui.QPen(overlay_color, 2))
                painter.setBrush(QtGui.QColor(0, 180, 255, 40))
                painter.drawRect(rect_widget)
                handle_size = 8.0
                painter.setBrush(overlay_color)
                for point in self._rect_corner_points(rect_widget):
                    handle_rect = QtCore.QRectF(
                        point.x() - handle_size / 2.0,
                        point.y() - handle_size / 2.0,
                        handle_size,
                        handle_size,
                    )
                    painter.drawRect(handle_rect)
            if self._dragging and not self._drag_rect_widget.isNull():
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1, QtCore.Qt.DashLine))
                painter.setBrush(QtGui.QColor(255, 255, 255, 40))
                painter.drawRect(self._drag_rect_widget)
        painter.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_scaled_pixmap()
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.LeftButton or self._scaled_pixmap.isNull():
            super().mousePressEvent(event)
            return
        pos = self._event_pos(event)
        if not self._pixmap_area().contains(pos):
            super().mousePressEvent(event)
            return
        self._dragging = True
        self._drag_start = pos
        self._drag_current = pos
        self._drag_rect_widget = QtCore.QRectF(pos, pos)
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if not self._dragging:
            super().mouseMoveEvent(event)
            return
        pos = self._event_pos(event)
        self._drag_current = pos
        self._drag_rect_widget = QtCore.QRectF(self._drag_start, self._drag_current).normalized()
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if not self._dragging:
            super().mouseReleaseEvent(event)
            return
        pos = self._event_pos(event)
        self._dragging = False
        self._drag_current = pos
        drag_rect = QtCore.QRectF(self._drag_start, self._drag_current).normalized()
        self._drag_rect_widget = QtCore.QRectF()
        if drag_rect.width() < 3.0 or drag_rect.height() < 3.0:
            self.update()
            return
        rect_img = self._widget_rect_to_image(drag_rect)
        if rect_img.width() < 1.0 or rect_img.height() < 1.0:
            self.update()
            return
        self._crop_rect = rect_img
        self.cropChanged.emit(QtCore.QRectF(self._crop_rect))
        self.update()

    def _pixmap_area(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            self._offset.x(),
            self._offset.y(),
            float(self._scaled_pixmap.width()),
            float(self._scaled_pixmap.height()),
        )

    def _rect_corner_points(self, rect: QtCore.QRectF) -> List[QtCore.QPointF]:
        return [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomRight(),
            rect.bottomLeft(),
        ]

    def _image_rect_to_widget(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        top_left = self._image_to_widget(rect.topLeft())
        bottom_right = self._image_to_widget(rect.bottomRight())
        return QtCore.QRectF(top_left, bottom_right).normalized()

    def _widget_rect_to_image(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        rect = rect.normalized()
        top_left = self._widget_to_image(rect.topLeft())
        bottom_right = self._widget_to_image(rect.bottomRight())
        return self._clip_rect_to_image(QtCore.QRectF(top_left, bottom_right))

    def _image_to_widget(self, point: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF(
            self._offset.x() + point.x() * self._scale,
            self._offset.y() + point.y() * self._scale,
        )

    def _widget_to_image(self, point: QtCore.QPointF) -> QtCore.QPointF:
        if self._scale <= 0.0:
            return QtCore.QPointF(0.0, 0.0)
        x = (point.x() - self._offset.x()) / self._scale
        y = (point.y() - self._offset.y()) / self._scale
        width = float(self._image_size.width())
        height = float(self._image_size.height())
        return QtCore.QPointF(
            min(max(x, 0.0), width),
            min(max(y, 0.0), height),
        )

    @staticmethod
    def _event_pos(event: QtGui.QMouseEvent) -> QtCore.QPointF:
        if hasattr(event, "position"):
            return event.position()
        return QtCore.QPointF(float(event.x()), float(event.y()))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, api: ProcessingAPI) -> None:
        super().__init__()
        self.api = api
        self.current_language = "en"
        self.translations: Dict[str, str] = {}
        self.current_preview_path: Optional[str] = None
        self.last_before_image: Optional[np.ndarray] = None
        self.last_after_image: Optional[np.ndarray] = None

        self._load_translations("en")
        self._build_ui()
        self._create_menus()
        self._apply_translations()
        self.status_label.setText(self._t("status_ready", "Ready."))

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------
    def _load_translations(self, code: str) -> None:
        base = _load_language_file("en")
        if code == "en":
            self.translations = base
        else:
            try:
                override = _load_language_file(code)
            except FileNotFoundError:
                override = {}
            merged = base.copy()
            merged.update(override)
            self.translations = merged
        self.current_language = code

    def _t(self, key: str, fallback: str = "") -> str:
        return self.translations.get(key, fallback)

    def _apply_translations(self) -> None:
        self.setWindowTitle("Neg2Posi")
        self.left_group_box.setTitle(self._t("left_panel_header", "Controls"))
        self.input_label.setText(self._t("input_label", "Input:"))
        self.output_label.setText(self._t("output_label", "Output Directory:"))
        self.select_file_btn.setText(self._t("select_file", "Select File"))
        self.select_folder_btn.setText(self._t("select_folder", "Select Folder"))
        self.choose_output_btn.setText(self._t("choose_output", "Choose Output Dir"))
        self.preview_btn.setText(self._t("preview", "Preview"))
        self.process_btn.setText(self._t("process", "Process"))
        self.edit_crop_btn.setText(self._t("edit_crop_button", "Edit Crop"))
        self.film_type_group_box.setTitle(self._t("film_type_label", "Processing Mode:"))
        self.color_radio.setText(self._t("film_type_color", "Color Negative"))
        self.bw_radio.setText(self._t("film_type_bw", "B&W Negative"))
        self.before_label.setText(self._t("label_before", "Before"))
        self.after_label.setText(self._t("label_after", "After"))
        self.status_label.setText(self._t("status_ready", "Ready."))

        language_menu = self.menuBar().findChild(QtWidgets.QMenu, "menu_language")
        if language_menu is not None:
            language_menu.setTitle(self._t("menu_language", "Language"))
            self.lang_actions["en"].setText(self._t("menu_language_english", "English"))
            self.lang_actions["zh"].setText(self._t("menu_language_chinese", "中文"))
            self.lang_actions["ja"].setText(self._t("menu_language_japanese", "日本語"))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left controls
        self.left_group_box = QtWidgets.QGroupBox()
        left_layout = QtWidgets.QVBoxLayout(self.left_group_box)
        splitter.addWidget(self.left_group_box)
        splitter.setStretchFactor(0, 0)

        self.film_type_group_box = QtWidgets.QGroupBox()
        film_layout = QtWidgets.QHBoxLayout(self.film_type_group_box)
        self.color_radio = QtWidgets.QRadioButton()
        self.bw_radio = QtWidgets.QRadioButton()
        self.color_radio.setChecked(True)
        film_layout.addWidget(self.color_radio)
        film_layout.addWidget(self.bw_radio)
        left_layout.addWidget(self.film_type_group_box)

        self.input_label = QtWidgets.QLabel()
        left_layout.addWidget(self.input_label)
        self.input_path_edit = QtWidgets.QLineEdit()
        self.input_path_edit.setReadOnly(True)
        left_layout.addWidget(self.input_path_edit)

        input_button_row = QtWidgets.QHBoxLayout()
        self.select_file_btn = QtWidgets.QPushButton()
        self.select_folder_btn = QtWidgets.QPushButton()
        input_button_row.addWidget(self.select_file_btn)
        input_button_row.addWidget(self.select_folder_btn)
        left_layout.addLayout(input_button_row)

        self.output_label = QtWidgets.QLabel()
        left_layout.addWidget(self.output_label)
        self.output_path_edit = QtWidgets.QLineEdit()
        self.output_path_edit.setReadOnly(True)
        left_layout.addWidget(self.output_path_edit)
        self.choose_output_btn = QtWidgets.QPushButton()
        left_layout.addWidget(self.choose_output_btn)

        self.preview_btn = QtWidgets.QPushButton()
        self.process_btn = QtWidgets.QPushButton()
        self.edit_crop_btn = QtWidgets.QPushButton()
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.preview_btn)
        button_row.addWidget(self.process_btn)
        left_layout.addLayout(button_row)
        left_layout.addWidget(self.edit_crop_btn)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        # Right preview area
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(1, 1)

        titles_layout = QtWidgets.QHBoxLayout()
        self.before_label = QtWidgets.QLabel()
        self.after_label = QtWidgets.QLabel()
        titles_layout.addWidget(self.before_label)
        titles_layout.addStretch(1)
        titles_layout.addWidget(self.after_label)
        right_layout.addLayout(titles_layout)

        preview_layout = QtWidgets.QHBoxLayout()
        self.before_image_view = CropPreview()
        self.after_image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.after_image_label.setMinimumSize(200, 200)
        self.after_image_label.setFrameShape(QtWidgets.QFrame.Box)
        preview_layout.addWidget(self.before_image_view, 1)
        preview_layout.addWidget(self.after_image_label, 1)
        right_layout.addLayout(preview_layout)

        # Signal connections
        self.select_file_btn.clicked.connect(self._choose_file)
        self.select_folder_btn.clicked.connect(self._choose_folder)
        self.choose_output_btn.clicked.connect(self._choose_output)
        self.preview_btn.clicked.connect(self._handle_preview)
        self.process_btn.clicked.connect(self._handle_process)
        self.edit_crop_btn.clicked.connect(self._open_crop_dialog)
        self.before_image_view.cropChanged.connect(self._handle_manual_crop_rect)

    def _create_menus(self) -> None:
        menu = self.menuBar().addMenu("")
        menu.setObjectName("menu_language")
        self.lang_actions: Dict[str, QtGui.QAction] = {}
        for code in ["en", "zh", "ja"]:
            action = QtGui.QAction(self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, c=code: self._switch_language(c))
            self.lang_actions[code] = action
            menu.addAction(action)
        self.lang_actions["en"].setChecked(True)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _switch_language(self, code: str) -> None:
        if code == self.current_language:
            return
        try:
            self._load_translations(code)
        except FileNotFoundError:
            self._load_translations("en")
        self._apply_translations()
        for c, action in self.lang_actions.items():
            action.setChecked(c == self.current_language)

    def _choose_file(self) -> None:
        caption = self._t("dialog_open_file", "Select negative")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption)
        if path:
            self.input_path_edit.setText(path)
            self.current_preview_path = path
            self.status_label.setText(self._t("status_ready", "Ready."))

    def _choose_folder(self) -> None:
        caption = self._t("dialog_open_folder", "Select folder")
        path = QtWidgets.QFileDialog.getExistingDirectory(self, caption)
        if path:
            self.input_path_edit.setText(path)
            self.current_preview_path = None
            self.status_label.setText(self._t("status_ready", "Ready."))

    def _choose_output(self) -> None:
        caption = self._t("dialog_open_output", "Select output directory")
        path = QtWidgets.QFileDialog.getExistingDirectory(self, caption)
        if path:
            self.output_path_edit.setText(path)

    def _selected_film_type(self) -> str:
        return "bw" if self.bw_radio.isChecked() else "color"

    def _collect_input_paths(self) -> Tuple[Path, List[str]]:
        in_path = self.input_path_edit.text().strip()
        if not in_path:
            raise FileNotFoundError(self._t("status_no_images", "No supported image files found."))
        path_obj = Path(in_path)
        if path_obj.is_file():
            return path_obj, [str(path_obj)]
        if path_obj.is_dir():
            files = [
                str(p)
                for p in sorted(path_obj.iterdir())
                if p.is_file() and p.suffix.lower() in self.api.allowed_extensions
            ]
            if not files:
                raise FileNotFoundError(self._t("status_no_images", "No supported image files found."))
            return path_obj, files
        raise FileNotFoundError(self._t("status_no_images", "No supported image files found."))

    def _handle_preview(self) -> None:
        try:
            _base_path, paths = self._collect_input_paths()
        except FileNotFoundError as exc:
            self.status_label.setText(str(exc))
            return
        film_type = self._selected_film_type()
        target = paths[0]
        try:
            before, after = self.api.preview_images(target, film_type)
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(
                self._t("status_preview_failed", "Preview failed: {}" ).format(exc)
            )
            return
        self.current_preview_path = target
        self._update_preview(before, after)
        self.status_label.setText(
            self._t("status_preview_generated", "Preview generated for {} image(s)." ).format(len(paths))
        )

    def _update_preview(self, before: np.ndarray, after: np.ndarray) -> None:
        self.last_before_image = before
        self.last_after_image = after
        crop_rect = self._current_crop_rect()
        self.before_image_view.set_image(before, crop_rect)
        self._refresh_after_label()

    def _refresh_after_label(self) -> None:
        if self.last_after_image is None:
            self.after_image_label.clear()
            return
        after_pixmap = numpy_to_qpixmap(self.last_after_image)
        self.after_image_label.setPixmap(
            after_pixmap.scaled(
                self.after_image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def _current_crop_rect(self) -> QtCore.QRectF:
        if self.last_before_image is None:
            return QtCore.QRectF()
        height, width = self.last_before_image.shape[:2]
        default_rect = QtCore.QRectF(0.0, 0.0, float(width), float(height))
        if self.current_preview_path is None:
            return default_rect
        stored_pts = self.api.user_crops.get(self.current_preview_path)
        if not stored_pts:
            return default_rect
        pts = np.array(stored_pts, dtype=np.float32)
        if pts.size == 0:
            return default_rect
        x_min = float(np.clip(pts[:, 0].min(), 0.0, width))
        x_max = float(np.clip(pts[:, 0].max() + 1.0, 0.0, width))
        y_min = float(np.clip(pts[:, 1].min(), 0.0, height))
        y_max = float(np.clip(pts[:, 1].max() + 1.0, 0.0, height))
        if x_max - x_min <= 0.0 or y_max - y_min <= 0.0:
            return default_rect
        return QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min)

    def _handle_manual_crop_rect(self, rect: QtCore.QRectF) -> None:
        if self.current_preview_path is None or self.last_before_image is None:
            return
        height, width = self.last_before_image.shape[:2]
        rect = rect.normalized()
        x0 = max(0.0, min(rect.x(), float(width)))
        y0 = max(0.0, min(rect.y(), float(height)))
        x1 = max(0.0, min(rect.x() + rect.width(), float(width)))
        y1 = max(0.0, min(rect.y() + rect.height(), float(height)))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            return
        new_margins = {
            "left": x0,
            "top": y0,
            "right": max(0.0, float(width) - x1),
            "bottom": max(0.0, float(height) - y1),
        }
        angle = float(self.api.user_angles.get(self.current_preview_path, 0.0))
        try:
            before, after = self.api.apply_crop_editor(
                self.current_preview_path,
                new_margins,
                angle,
                self._selected_film_type(),
            )
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(
                self._t("status_preview_failed", "Preview failed: {}" ).format(exc)
            )
            return
        self._update_preview(before, after)
        self.status_label.setText(
            self._t("status_preview_generated", "Preview generated for {} image(s)." ).format(1)
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_after_label()

    def _handle_process(self) -> None:
        try:
            base_path, _ = self._collect_input_paths()
        except FileNotFoundError as exc:
            self.status_label.setText(str(exc))
            return
        out_dir = self.output_path_edit.text().strip()
        if not out_dir:
            out_dir = str(base_path.parent if base_path.is_file() else base_path)
            self.output_path_edit.setText(out_dir)
        film_type = self._selected_film_type()
        try:
            self.api.process_path(
                str(base_path),
                out_dir,
                film_type,
                recurse=base_path.is_dir(),
                crop_dict=self.api.user_crops,
            )
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(
                self._t("status_processing_failed", "Processing failed: {}" ).format(exc)
            )
            return
        self.status_label.setText(self._t("status_processing_complete", "Processing finished."))

    def _open_crop_dialog(self) -> None:
        if self.current_preview_path is None:
            self.status_label.setText(self._t("status_no_images", "No supported image files found."))
            return
        if self.last_before_image is None:
            try:
                before, after = self.api.preview_images(self.current_preview_path, self._selected_film_type())
            except Exception as exc:  # noqa: BLE001
                self.status_label.setText(
                    self._t("status_preview_failed", "Preview failed: {}" ).format(exc)
                )
                return
            self._update_preview(before, after)
        margins, angle = self.api.get_crop_settings(self.current_preview_path)
        dialog = CropDialog(self, self.translations, margins, angle, self.last_before_image.shape[:2])
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            new_margins, new_angle = dialog.values()
            try:
                before, after = self.api.apply_crop_editor(
                    self.current_preview_path,
                    new_margins,
                    new_angle,
                    self._selected_film_type(),
                )
            except Exception as exc:  # noqa: BLE001
                self.status_label.setText(
                    self._t("status_preview_failed", "Preview failed: {}" ).format(exc)
                )
                return
            self._update_preview(before, after)
            self.status_label.setText(
                self._t("status_preview_generated", "Preview generated for {} image(s)." ).format(1)
            )


def run_qt_app(api: ProcessingAPI) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(api)
    window.resize(1200, 700)
    window.show()
    app.exec()
