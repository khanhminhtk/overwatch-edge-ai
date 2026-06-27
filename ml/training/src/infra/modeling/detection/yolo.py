from collections.abc import Iterable
from logging import getLogger
from pathlib import Path
import shutil
import tempfile
from typing import Any

from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.domain.value_objet.config_yolo import YoloConfig, YoloArgsAdapter

LOGGER = getLogger(__name__)


class YoloTrainer:
    def __init__(self, config: YoloConfig) -> None:
        self.config = config
        self._model: YOLO | None = None
        self._metrics = None
        self._result = None

    def load(self, finetune: bool) -> None:
        model_source = self._resolve_model_source(finetune=finetune)
        self._model = YOLO(model=model_source, task=self.config.model.task)

    def load_model_best_weights(self, path: str) -> None:
        self._model = YOLO(model=path, task=self.config.model.task)

    def train(self) -> Any:
        if self._model is None:
            self.load(finetune=False)

        if self._model is None:
            raise RuntimeError("Failed to initialize YOLO model before training.")

        train_kwargs = YoloArgsAdapter.to_train_kwargs(self.config)
        self._result = self._model.train(**train_kwargs)
        return self._result
    
    def validate(self, **kwargs: Any) -> dict[str, float]:
        if self._model is None:
            self.load(finetune=False)

        if self._model is None:
            raise RuntimeError("Failed to initialize YOLO model before validation.")

        val_kwargs = YoloArgsAdapter.to_val_kwargs(self.config, **kwargs)
        self._metrics = self._model.val(**val_kwargs)
        return {
            "mAP50": float(self._metrics.box.map50),
            "mAP50-95": float(self._metrics.box.map),
            "precision": float(self._metrics.box.mp),
            "recall": float(self._metrics.box.mr),
        }

    def _resolve_model_source(self, finetune: bool) -> str:
        if finetune:
            checkpoint = self._project_dir() / self.config.save.run_name / "weights" / "last.pt"
            if checkpoint.exists():
                return str(checkpoint)
        weights = self.config.model.weights
        if not self.config.model.pretrained and weights.endswith(".pt"):
            raise ValueError(
                "model.pretrained=false expects model.weights to point to a model architecture yaml, not a .pt checkpoint."
            )
        return weights

    def predict(
        self,
        source: str | Path | list[str | Path],
        conf: float = 0.5,
        save: bool = False,
        device: str = "cpu",
        stream: bool = False,
        verbose: bool = False,
        **kwargs: Any
    ) -> list[Results] | Iterable[Results]:
        if self._model is None:
            self.load(finetune=False)
        if self._model is None:
            raise RuntimeError("Failed to initialize YOLO model before prediction.")

        return self._model.predict(
            source=source,
            conf=conf,
            save=save,
            device=device,
            stream=stream,
            verbose=verbose,
            **kwargs
        )
    
    def get_boxes(self, results: list[Results] | Iterable[Results]) -> list[dict[str, list[float] | float | int | str]]:
        all_detections: list[dict[str, list[float] | float | int | str]] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                }
                all_detections.append(detection)
        
        return all_detections
    
    def save(self, save_path: str | Path) -> str:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            self.load(finetune=False)
        if self._model is None:
            raise RuntimeError("Failed to initialize YOLO model before saving.")
        self._model.save(str(save_path_obj))
        LOGGER.info("Model saved to %s", save_path_obj)
        return str(save_path_obj)
    
    @property
    def model(self) -> YOLO | None:
        return self._model
    
    @property
    def result(self) -> Any:
        return self._result
    
    @property
    def metrics(self) -> Any:
        return self._metrics
    
    @property
    def best_weight_path(self) -> str:
        return str(self._project_dir() / self.config.save.run_name / "weights" / "best.pt")

    def _project_dir(self) -> Path:
        return Path(self.config.save.project).expanduser().resolve()

    def _export_to_path(self, export_path: str | Path, fmt: str) -> str:
        if self._model is None:
            self.load(finetune=False)
        if self._model is None:
            raise RuntimeError(f"Failed to initialize YOLO model before {fmt.upper()} export.")

        export_path_obj = Path(export_path).expanduser().resolve()
        export_path_obj.parent.mkdir(parents=True, exist_ok=True)

        export_dir = export_path_obj.parent
        run_name = export_path_obj.stem
        original_pt_path = getattr(self._model.model, "pt_path", None)
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        try:
            if original_pt_path:
                original_pt_path_obj = Path(original_pt_path).expanduser().resolve()
                temp_dir = tempfile.TemporaryDirectory(prefix="yolo-export-", dir=str(export_dir))
                temp_pt_path = Path(temp_dir.name) / original_pt_path_obj.name
                shutil.copy2(original_pt_path_obj, temp_pt_path)
                self._model.model.pt_path = str(temp_pt_path)

            exported = self._model.export(
                format=fmt,
                project=str(export_dir),
                name=run_name,
                exist_ok=True,
                device="cpu",
                dynamic=True,
                end2end=False,
                simplify=False,
            )
            produced = Path(str(exported)).expanduser().resolve()
            if not produced.exists():
                raise RuntimeError(f"Cannot locate exported .{fmt} artifact: {produced}")
            if produced.resolve() != export_path_obj:
                shutil.move(str(produced), str(export_path_obj))
            return str(export_path_obj)
        finally:
            if original_pt_path:
                self._model.model.pt_path = original_pt_path
            if temp_dir is not None:
                temp_dir.cleanup()

    def export_onnx(self, export_path: str | Path) -> str:
        exported_path = self._export_to_path(
            export_path=export_path, 
            fmt="onnx"
        )
        LOGGER.info("Model exported to ONNX format at %s", exported_path)
        return exported_path
    
    def export_tensorrt(self, export_path: str | Path) -> str:
        exported_path = self._export_to_path(export_path=export_path, fmt="engine")
        LOGGER.info("Model exported to TensorRT format at %s", exported_path)
        return exported_path
