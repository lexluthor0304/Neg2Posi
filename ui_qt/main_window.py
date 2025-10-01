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


class CropGraphicsView(QtWidgets.QGraphicsView):
    selectionChanged = QtCore.Signal(dict)

    def __init__(self, image: np.ndarray) -> None:
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap = numpy_to_qpixmap(image)
        self._pixmap_item = self._scene.addPixmap(self._pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())

        pen = QtGui.QPen(QtGui.QColor(255, 99, 71))
        pen.setWidth(2)
        brush = QtGui.QBrush(QtGui.QColor(255, 99, 71, 60))
        self._selection_item = self._scene.addRect(QtCore.QRectF(), pen, brush)
        self._selection_item.setVisible(False)
        self._selection_rect = QtCore.QRectF()

        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setMouseTracking(True)

        self._dragging = False
        self._drawing = False
        self._drag_offset = QtCore.QPointF()
        self._draw_start = QtCore.QPointF()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._pixmap.isNull():
            self.fitInView(self._pixmap_item, QtCore.Qt.KeepAspectRatio)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            if self._selection_rect.contains(scene_pos):
                self._dragging = True
                self._drag_offset = scene_pos - self._selection_rect.topLeft()
            else:
                self._drawing = True
                self._draw_start = self._clamp_to_scene(scene_pos)
                self._selection_rect = QtCore.QRectF(self._draw_start, self._draw_start)
                self._update_selection_item()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._dragging:
            self._move_selection(scene_pos)
        elif self._drawing:
            end_pos = self._clamp_to_scene(scene_pos)
            rect = QtCore.QRectF(self._draw_start, end_pos).normalized()
            self._selection_rect = rect
            self._update_selection_item()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            if self._dragging or self._drawing:
                self._selection_rect = self._selection_rect.intersected(self._scene.sceneRect())
                if self._selection_rect.width() < 1 or self._selection_rect.height() < 1:
                    self._selection_rect = self._scene.sceneRect()
                self._update_selection_item()
                if self._selection_item.isVisible():
                    self.selectionChanged.emit(self.current_margins())
            self._dragging = False
            self._drawing = False
        super().mouseReleaseEvent(event)

    def _move_selection(self, scene_pos: QtCore.QPointF) -> None:
        rect = QtCore.QRectF(self._selection_rect)
        top_left = scene_pos - self._drag_offset
        rect.moveTopLeft(top_left)
        rect = rect.intersected(self._scene.sceneRect())
        self._selection_rect = rect
        self._update_selection_item()

    def _clamp_to_scene(self, pos: QtCore.QPointF) -> QtCore.QPointF:
        rect = self._scene.sceneRect()
        x = min(max(pos.x(), rect.left()), rect.right())
        y = min(max(pos.y(), rect.top()), rect.bottom())
        return QtCore.QPointF(x, y)

    def _update_selection_item(self) -> None:
        rect = self._selection_rect.intersected(self._scene.sceneRect())
        if rect.width() < 1 or rect.height() < 1:
            self._selection_item.setVisible(False)
            return
        self._selection_item.setRect(rect)
        self._selection_item.setVisible(True)

    def set_selection_from_margins(self, margins: Dict[str, float], emit: bool = False) -> None:
        width = self._pixmap.width()
        height = self._pixmap.height()
        left = max(0.0, margins.get("left", 0.0))
        right = max(0.0, margins.get("right", 0.0))
        top = max(0.0, margins.get("top", 0.0))
        bottom = max(0.0, margins.get("bottom", 0.0))
        rect = QtCore.QRectF(
            left,
            top,
            max(0.0, width - left - right),
            max(0.0, height - top - bottom),
        )
        if rect.width() <= 0 or rect.height() <= 0:
            rect = self._scene.sceneRect()
        self._selection_rect = rect.intersected(self._scene.sceneRect())
        self._update_selection_item()
        if emit and self._selection_item.isVisible():
            self.selectionChanged.emit(self.current_margins())

    def current_margins(self) -> Dict[str, float]:
        rect = self._selection_rect.intersected(self._scene.sceneRect())
        width = self._pixmap.width()
        height = self._pixmap.height()
        return {
            "left": max(0.0, rect.left()),
            "top": max(0.0, rect.top()),
            "right": max(0.0, width - rect.right()),
            "bottom": max(0.0, height - rect.bottom()),
        }


class CropDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        translations: Dict[str, str],
        margins: Dict[str, float],
        angle: float,
        image: np.ndarray,
    ) -> None:
        super().__init__(parent)
        self._t = translations
        self.setWindowTitle(self._t.get("crop_editor_title", "Edit Crop"))
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        self.view = CropGraphicsView(image)
        layout.addWidget(self.view, stretch=1)

        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QFormLayout(controls_widget)
        layout.addWidget(controls_widget)

        h, w = image.shape[:2]
        self._syncing = False

        self.left_spin = QtWidgets.QSpinBox()
        self.left_spin.setRange(0, max(0, w - 1))

        self.right_spin = QtWidgets.QSpinBox()
        self.right_spin.setRange(0, max(0, w - 1))

        self.top_spin = QtWidgets.QSpinBox()
        self.top_spin.setRange(0, max(0, h - 1))

        self.bottom_spin = QtWidgets.QSpinBox()
        self.bottom_spin.setRange(0, max(0, h - 1))

        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(-180.0, 180.0)
        self.angle_spin.setDecimals(2)
        self.angle_spin.setSingleStep(0.5)
        self.angle_spin.setValue(float(angle))

        controls_layout.addRow(self._t.get("crop_margin_left", "Left (px)"), self.left_spin)
        controls_layout.addRow(self._t.get("crop_margin_right", "Right (px)"), self.right_spin)
        controls_layout.addRow(self._t.get("crop_margin_top", "Top (px)"), self.top_spin)
        controls_layout.addRow(self._t.get("crop_margin_bottom", "Bottom (px)"), self.bottom_spin)
        controls_layout.addRow(self._t.get("crop_rotation_angle", "Rotation Angle (°)"), self.angle_spin)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.button(QtWidgets.QDialogButtonBox.Ok).setText(self._t.get("dialog_crop_apply", "Apply"))
        button_box.button(QtWidgets.QDialogButtonBox.Cancel).setText(self._t.get("dialog_crop_cancel", "Cancel"))
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.left_spin.valueChanged.connect(self._spins_changed)
        self.right_spin.valueChanged.connect(self._spins_changed)
        self.top_spin.valueChanged.connect(self._spins_changed)
        self.bottom_spin.valueChanged.connect(self._spins_changed)
        self.view.selectionChanged.connect(self._view_selection_changed)

        self._apply_margins_to_ui(margins)

    def _apply_margins_to_ui(self, margins: Dict[str, float]) -> None:
        self._syncing = True
        self.left_spin.setValue(int(round(margins.get("left", 0.0))))
        self.right_spin.setValue(int(round(margins.get("right", 0.0))))
        self.top_spin.setValue(int(round(margins.get("top", 0.0))))
        self.bottom_spin.setValue(int(round(margins.get("bottom", 0.0))))
        self._syncing = False
        self.view.set_selection_from_margins(margins)

    def _spins_changed(self) -> None:
        if self._syncing:
            return
        margins = {
            "left": float(self.left_spin.value()),
            "right": float(self.right_spin.value()),
            "top": float(self.top_spin.value()),
            "bottom": float(self.bottom_spin.value()),
        }
        self.view.set_selection_from_margins(margins, emit=True)

    def _view_selection_changed(self, margins: Dict[str, float]) -> None:
        self._syncing = True
        self.left_spin.setValue(int(round(margins.get("left", 0.0))))
        self.right_spin.setValue(int(round(margins.get("right", 0.0))))
        self.top_spin.setValue(int(round(margins.get("top", 0.0))))
        self.bottom_spin.setValue(int(round(margins.get("bottom", 0.0))))
        self._syncing = False

    def values(self) -> Tuple[Dict[str, float], float]:
        return self.view.current_margins(), float(self.angle_spin.value())


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
        self.before_image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.after_image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.before_image_label.setMinimumSize(200, 200)
        self.after_image_label.setMinimumSize(200, 200)
        self.before_image_label.setFrameShape(QtWidgets.QFrame.Box)
        self.after_image_label.setFrameShape(QtWidgets.QFrame.Box)
        preview_layout.addWidget(self.before_image_label, 1)
        preview_layout.addWidget(self.after_image_label, 1)
        right_layout.addLayout(preview_layout)

        # Signal connections
        self.select_file_btn.clicked.connect(self._choose_file)
        self.select_folder_btn.clicked.connect(self._choose_folder)
        self.choose_output_btn.clicked.connect(self._choose_output)
        self.preview_btn.clicked.connect(self._handle_preview)
        self.process_btn.clicked.connect(self._handle_process)
        self.edit_crop_btn.clicked.connect(self._open_crop_dialog)

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
        self._refresh_preview_labels()

    def _refresh_preview_labels(self) -> None:
        if self.last_before_image is None or self.last_after_image is None:
            return
        before_pixmap = numpy_to_qpixmap(self.last_before_image)
        after_pixmap = numpy_to_qpixmap(self.last_after_image)
        self.before_image_label.setPixmap(
            before_pixmap.scaled(
                self.before_image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )
        self.after_image_label.setPixmap(
            after_pixmap.scaled(
                self.after_image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_preview_labels()

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
        dialog = CropDialog(self, self.translations, margins, angle, self.last_before_image)
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
