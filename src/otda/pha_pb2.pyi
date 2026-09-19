from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class AsteroidFeatures(_message.Message):
    __slots__ = ("e", "q", "h", "is_neo", "class_code", "moid")
    E_FIELD_NUMBER: _ClassVar[int]
    Q_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    IS_NEO_FIELD_NUMBER: _ClassVar[int]
    CLASS_CODE_FIELD_NUMBER: _ClassVar[int]
    MOID_FIELD_NUMBER: _ClassVar[int]
    e: float
    q: float
    h: float
    is_neo: bool
    class_code: int
    moid: float
    def __init__(self, e: _Optional[float] = ..., q: _Optional[float] = ..., h: _Optional[float] = ..., is_neo: bool = ..., class_code: _Optional[int] = ..., moid: _Optional[float] = ...) -> None: ...

class PredictionResponse(_message.Message):
    __slots__ = ("is_pha", "probability", "explanation")
    IS_PHA_FIELD_NUMBER: _ClassVar[int]
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    EXPLANATION_FIELD_NUMBER: _ClassVar[int]
    is_pha: bool
    probability: float
    explanation: str
    def __init__(self, is_pha: bool = ..., probability: _Optional[float] = ..., explanation: _Optional[str] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "model_loaded")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_LOADED_FIELD_NUMBER: _ClassVar[int]
    status: str
    model_loaded: bool
    def __init__(self, status: _Optional[str] = ..., model_loaded: bool = ...) -> None: ...
