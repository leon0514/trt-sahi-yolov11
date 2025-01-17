from __future__ import annotations
import numpy
import typing
__all__ = ['Box', 'TrtSahiYolo', 'YOLOV11', 'YOLOV5', 'YOLOV8', 'YoloType']
class Box:
    bottom: float
    class_label: int
    confidence: float
    left: float
    right: float
    top: float
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
class TrtSahiYolo:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, model_path: str, gpu_id: YoloType, yolo_type: int) -> None:
        ...
    def autoSliceForward(self, image: numpy.ndarray) -> list[Box]:
        ...
    def manualSliceForward(self, image: numpy.ndarray, width: int, height: int, xratio: float, yratio: float) -> list[Box]:
        ...
    @property
    def valid(self) -> bool:
        ...
class YoloType:
    """
    Members:
    
      YOLOV5
    
      YOLOV8
    
      YOLOV11
    """
    YOLOV11: typing.ClassVar[YoloType]  # value = <YoloType.YOLOV11: 2>
    YOLOV5: typing.ClassVar[YoloType]  # value = <YoloType.YOLOV5: 0>
    YOLOV8: typing.ClassVar[YoloType]  # value = <YoloType.YOLOV8: 1>
    __members__: typing.ClassVar[dict[str, YoloType]]  # value = {'YOLOV5': <YoloType.YOLOV5: 0>, 'YOLOV8': <YoloType.YOLOV8: 1>, 'YOLOV11': <YoloType.YOLOV11: 2>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
YOLOV11: YoloType  # value = <YoloType.YOLOV11: 2>
YOLOV5: YoloType  # value = <YoloType.YOLOV5: 0>
YOLOV8: YoloType  # value = <YoloType.YOLOV8: 1>
