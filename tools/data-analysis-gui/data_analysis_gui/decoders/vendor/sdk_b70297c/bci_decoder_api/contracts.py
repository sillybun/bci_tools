# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/bci_decoder_api/contracts.py
# original git blob: 9e53bec126fc83f48194b40e57d90611b24eb5a4
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Decoder 集成层与模型 Core 共用的公开、通信框架无关的数据合同。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import numpy as np

from .specs import FeatureSpec, FeedbackSpec, validate_values_against_spec

if TYPE_CHECKING:
    from .artifact import ModelArtifact


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
_REASON_CODE_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$"
)


def frozen_arrays(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Return read-only copies so checkpoints and artifacts are deeply immutable."""

    copied: dict[str, np.ndarray] = {}
    for key, value in values.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{key} must be a NumPy array")
        array = np.array(value, copy=True)
        if array.dtype.hasobject:
            raise TypeError(f"{key} cannot use an object dtype")
        array.setflags(write=False)
        copied[str(key)] = array
    return MappingProxyType(copied)


def _finite_metrics(values: Mapping[str, float]) -> Mapping[str, float]:
    copied: dict[str, float] = {}
    for key, value in values.items():
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"metric {key!r} must be finite")
        copied[str(key)] = scalar
    return MappingProxyType(copied)


@dataclass(frozen=True)
class FeatureTensor:
    """上游特征生产模块生成的一帧特征；Decoder update ID 由 Runtime 管理。"""

    # 当前逻辑更新使用的特征值；必须是非标量 CPU NumPy float32 数组，shape 与 Spec 一致。
    values: np.ndarray
    # 与 values 同 shape 的 bool 数组；True 表示对应元素有效，False 表示无效。
    valid_mask: np.ndarray
    # 特征的公共语义 Schema 标识；必须与 Artifact FeatureSpec 完全一致。
    schema_id: str
    # 本次部署的具体张量布局标识；必须与 Artifact FeatureSpec 完全一致。
    layout_id: str
    # 当前特征窗口结束时间，单位为纳秒；必须为非负数并在同一 Runtime 中严格递增。
    timestamp_ns: int
    # Producer 分配的帧编号；允许不提供。它用于追踪输入帧，不是 Decoder update ID。
    producer_frame_id: int | None = None

    @classmethod
    def from_spec(
        cls,
        spec: FeatureSpec,
        *,
        values: np.ndarray,
        valid_mask: np.ndarray,
        timestamp_ns: int,
        producer_frame_id: int | None = None,
    ) -> "FeatureTensor":
        """Bind producer values to an exact feature contract.

        This validates dtype/shape and copies schema/layout IDs from ``spec``.
        It does not compute, reorder, reshape, cast, or otherwise preprocess
        feature values.
        """

        validate_values_against_spec(values, spec, field_name="feature values")
        return cls(
            values=values,
            valid_mask=valid_mask,
            schema_id=spec.schema_id,
            layout_id=spec.layout_id,
            timestamp_ns=timestamp_ns,
            producer_frame_id=producer_frame_id,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray) or self.values.ndim == 0:
            raise TypeError("FeatureTensor.values must be a non-scalar NumPy array")
        if self.values.dtype != np.dtype("float32"):
            raise TypeError("FeatureTensor.values must use float32")
        if not isinstance(self.valid_mask, np.ndarray):
            raise TypeError("FeatureTensor.valid_mask must be a NumPy array")
        if self.valid_mask.dtype != np.dtype("bool"):
            raise TypeError("FeatureTensor.valid_mask must use bool")
        if self.valid_mask.shape != self.values.shape:
            raise ValueError("valid_mask must have the same shape as values")
        if not self.schema_id or not self.layout_id:
            raise ValueError("schema_id and layout_id are required")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.producer_frame_id is not None and self.producer_frame_id < 0:
            raise ValueError("producer_frame_id must be non-negative when provided")


@dataclass(frozen=True)
class FeedbackState:
    """与当前特征帧配套的外部任务反馈。"""

    # 外部反馈值；必须是非标量 CPU NumPy float32 数组，shape 与 FeedbackSpec 一致。
    values: np.ndarray
    # 反馈量的公共语义 Schema 标识；必须与 Artifact FeedbackSpec 完全一致。
    schema_id: str
    # 反馈量的具体张量布局标识；必须与 Artifact FeedbackSpec 完全一致。
    layout_id: str
    # 反馈对应的观测时间，单位为纳秒；不能晚于当前 feature，且不能超过最大允许 age。
    timestamp_ns: int

    @classmethod
    def from_spec(
        cls,
        spec: FeedbackSpec,
        *,
        values: np.ndarray,
        timestamp_ns: int,
    ) -> "FeedbackState":
        """Bind external feedback values to an exact feedback contract.

        This validates dtype/shape and copies schema/layout IDs from ``spec``;
        it does not transform or synchronize feedback values.
        """

        validate_values_against_spec(values, spec, field_name="feedback values")
        return cls(
            values=values,
            schema_id=spec.schema_id,
            layout_id=spec.layout_id,
            timestamp_ns=timestamp_ns,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray) or self.values.ndim == 0:
            raise TypeError("FeedbackState.values must be a non-scalar NumPy array")
        if self.values.dtype != np.dtype("float32"):
            raise TypeError("FeedbackState.values must use float32")
        if not self.schema_id or not self.layout_id or self.timestamp_ns < 0:
            raise ValueError(
                "feedback schema_id/layout_id are required and timestamp must be non-negative"
            )


@dataclass(frozen=True)
class DecoderRequest:
    """集成方提交给公开 ``DecoderRuntime.infer()`` 的单步请求。"""

    # 当前特征帧；Runtime 会校验 Spec、时间顺序和上游 sequence ID。
    features: FeatureTensor
    # 当前外部反馈；Artifact 不声明 FeedbackSpec 时必须为 None，否则每次请求都要提供。
    feedback: FeedbackState | None = None


@dataclass(frozen=True)
class CoreDecoderRequest:
    """SDK Runtime 完成公共校验后传给模型特有 Core 的请求。"""

    # 已通过 Runtime 公共合同校验的当前特征帧。
    features: FeatureTensor
    # 已通过 Runtime 公共合同校验的反馈；无反馈合同的模型为 None。
    feedback: FeedbackState | None
    # Runtime 为本次候选成功更新分配的编号；只有 Core 返回 OK 后才会被正式消耗。
    update_id: int

    def __post_init__(self) -> None:
        if self.update_id < 0:
            raise ValueError("update_id must be non-negative")


@dataclass(frozen=True)
class DecoderOutput:
    """Runtime 为一次成功推理生成的公开 Decoder 输出。"""

    # 模型输出 action 的独立副本；CPU NumPy float32，shape 与 Artifact ActionSpec 一致。
    action: np.ndarray
    # action 的公共语义 Schema 标识，由 Runtime 从 Artifact ActionSpec 填入。
    action_schema_id: str
    # action 的具体张量布局标识，由 Runtime 从 Artifact ActionSpec 填入。
    action_layout_id: str
    # 产生该 action 的输入 feature 时间戳，单位为纳秒。
    input_timestamp_ns: int
    # Runtime 成功更新编号；从 0 开始且只在成功提交 Core 状态后连续递增。
    update_id: int
    # 对应输入帧的 producer frame ID；输入未提供时为 None。
    producer_frame_id: int | None
    # 产生该 action 的具体 Artifact 参数版本，用于输出追溯。
    model_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, np.ndarray) or self.action.ndim == 0:
            raise TypeError("action must be a non-scalar NumPy array")
        if self.action.dtype != np.dtype("float32"):
            raise TypeError("action must use float32")
        if not self.action_schema_id or not self.action_layout_id or not self.model_version:
            raise ValueError("action schema/layout IDs and model_version are required")
        if self.input_timestamp_ns < 0 or self.update_id < 0:
            raise ValueError("output timestamp and update_id must be non-negative")


class InferenceStatus(str, Enum):
    """单步推理的粗粒度状态；具体失败原因由 diagnostics.reason_code 表达。"""

    # 推理成功，结果中必须包含 output/action。
    OK = "ok"
    # 特征 valid_mask、有限值或模型规定的数值域不满足要求。
    INVALID_FEATURES = "invalid_features"
    # 输入的 schema、layout、dtype 或 shape 与 Artifact 合同不一致。
    SCHEMA_MISMATCH = "schema_mismatch"
    # feature 时间戳或 producer_frame_id 重复、倒退。
    OUT_OF_ORDER = "out_of_order"
    # Artifact 要求 feedback，但本次请求没有提供。
    MISSING_FEEDBACK = "missing_feedback"
    # feedback 来自未来，或早于 Artifact 配置允许的最大 age。
    STALE_FEEDBACK = "stale_feedback"
    # 模型计算产生非有限值、奇异状态或其他可预期数值失败。
    NUMERICAL_ERROR = "numerical_error"
    # 失败不属于上述公共类别；必须给出模型命名空间 reason_code。
    MODEL_SPECIFIC_ERROR = "model_specific_error"


@dataclass(frozen=True)
class CoreInferenceDiagnostics:
    """模型 Core 返回的诊断信息；通用调用耗时由 Runtime 统一测量。"""

    # 本次输入中被判定为无效的 feature 元素数量；没有无效元素时为 0。
    invalid_feature_count: int = 0
    # 机器可读失败原因；使用 sdk.* 或模型命名空间，成功时必须为 None。
    reason_code: str | None = None
    # 面向人的诊断消息；None 表示没有附加消息，不得将文本内容作为程序控制合同。
    message: str | None = None
    # 模型或 Runtime 附加的有限数值 telemetry；构造后冻结为只读映射。
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.invalid_feature_count < 0:
            raise ValueError("diagnostic counts must be non-negative")
        if self.reason_code is not None and not _REASON_CODE_PATTERN.fullmatch(
            self.reason_code
        ):
            raise ValueError(
                "reason_code must be a namespaced lower_snake_case code "
                "such as 'sdk.schema_mismatch'"
            )
        object.__setattr__(self, "metrics", _finite_metrics(self.metrics))


@dataclass(frozen=True)
class InferenceDiagnostics:
    """SDK Runtime 返回给集成方的一次公开推理诊断。"""

    # DecoderRuntime.infer() 的端到端耗时，包含锁等待、公共检查、Core 和输出封装。
    inference_time_ns: int
    # infer_core() 调用边界耗时；请求在进入 Core 前被拒绝时为 None。
    core_inference_time_ns: int | None = None
    # 本次输入中被判定为无效的 feature 元素数量；没有无效元素时为 0。
    invalid_feature_count: int = 0
    # 机器可读失败原因；使用 sdk.* 或模型命名空间，成功时必须为 None。
    reason_code: str | None = None
    # 面向人的诊断消息；None 表示没有附加消息，不得将文本内容作为程序控制合同。
    message: str | None = None
    # 模型或 Runtime 附加的有限数值 telemetry；构造后冻结为只读映射。
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.inference_time_ns < 0
            or (
                self.core_inference_time_ns is not None
                and self.core_inference_time_ns < 0
            )
            or self.invalid_feature_count < 0
        ):
            raise ValueError("diagnostic counts and durations must be non-negative")
        if self.reason_code is not None and not _REASON_CODE_PATTERN.fullmatch(
            self.reason_code
        ):
            raise ValueError(
                "reason_code must be a namespaced lower_snake_case code "
                "such as 'sdk.schema_mismatch'"
            )
        object.__setattr__(self, "metrics", _finite_metrics(self.metrics))


@dataclass(frozen=True)
class CoreInferenceResult:
    """模型 Core 返回、尚未由 Runtime 添加公开输出身份的结果。"""

    # Core 推理状态；OK 与 action 存在严格对应，非 OK 表示本次状态不得提交。
    status: InferenceStatus
    # 候选 action；仅 status 为 OK 时存在，必须是 CPU NumPy float32 数组。
    action: np.ndarray | None
    # Core 发现的无效特征、精确失败原因和模型特有 telemetry；不包含通用调用耗时。
    diagnostics: CoreInferenceDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, CoreInferenceDiagnostics):
            raise TypeError(
                "CoreInferenceResult.diagnostics must be CoreInferenceDiagnostics"
            )
        if self.status is InferenceStatus.OK and self.action is None:
            raise ValueError("OK core results require an action")
        if self.status is not InferenceStatus.OK and self.action is not None:
            raise ValueError("failed core results must not contain an action")
        if (
            self.status is InferenceStatus.OK
            and self.diagnostics.reason_code is not None
        ):
            raise ValueError("OK core results must not contain a reason_code")
        if (
            self.status is InferenceStatus.MODEL_SPECIFIC_ERROR
            and self.diagnostics.reason_code is None
        ):
            raise ValueError("MODEL_SPECIFIC_ERROR core results require a reason_code")
        if self.action is not None:
            if not isinstance(self.action, np.ndarray) or self.action.ndim == 0:
                raise TypeError("core action must be a non-scalar NumPy array")
            if self.action.dtype != np.dtype("float32"):
                raise TypeError("core action must use float32")


@dataclass(frozen=True)
class InferenceResult:
    """``DecoderRuntime.infer()`` 返回给集成方的公开结果。"""

    # 公开推理状态；OK 与 output 存在严格对应，非 OK 不产生兜底 action。
    status: InferenceStatus
    # 成功时的可追溯 Decoder 输出；任何非 OK 状态下必须为 None。
    output: DecoderOutput | None
    # Runtime 端到端耗时、Core 耗时及其他诊断信息。
    diagnostics: InferenceDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, InferenceDiagnostics):
            raise TypeError("InferenceResult.diagnostics must be InferenceDiagnostics")
        if self.status is InferenceStatus.OK and self.output is None:
            raise ValueError("OK results require an output")
        if self.status is not InferenceStatus.OK and self.output is not None:
            raise ValueError("failed results must not contain an output")
        if (
            self.status is InferenceStatus.OK
            and self.diagnostics.reason_code is not None
        ):
            raise ValueError("OK results must not contain a reason_code")
        if (
            self.status is InferenceStatus.MODEL_SPECIFIC_ERROR
            and self.diagnostics.reason_code is None
        ):
            raise ValueError("MODEL_SPECIFIC_ERROR results require a reason_code")


@dataclass(frozen=True)
class RuntimeResetRequest:
    """集成方在会话边界提交给 Runtime/Core 的 reset 上下文。"""

    # 外部 reset 事件时间，单位为纳秒；不代表成功 feature 时间戳，不会推进推理顺序。
    timestamp_ns: int | None = None
    # reset 时的外部反馈；仅支持按反馈初始化的模型使用，其他情况保持为 None。
    feedback: FeedbackState | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ns is not None and self.timestamp_ns < 0:
            raise ValueError("reset timestamp must be non-negative")


@dataclass(frozen=True)
class CoreRuntimeState:
    """仅包含模型特有状态的 Core checkpoint。"""

    # 模型状态结构版本；用于拒绝 shape/语义不兼容的 checkpoint。
    state_schema_id: str
    # 模型特有状态数组；构造时深拷贝并设为只读，不包含公共 Runtime 顺序状态。
    arrays: Mapping[str, np.ndarray]
    # 模型特有的非负整数计数器；不包含 Runtime update ID 或 producer frame ID。
    counters: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state_schema_id:
            raise ValueError("core state_schema_id is required")
        if any(int(value) < 0 for value in self.counters.values()):
            raise ValueError("core-state counters must be non-negative")
        object.__setattr__(self, "arrays", frozen_arrays(self.arrays))
        object.__setattr__(
            self,
            "counters",
            MappingProxyType({str(key): int(value) for key, value in self.counters.items()}),
        )


@dataclass(frozen=True)
class RuntimeState:
    """包含模型 Core 状态和公共顺序状态的完整 Runtime checkpoint。"""

    # Decoder 算法家族标识；restore 时必须与目标 Runtime Artifact 完全一致。
    model_type: str
    # 具体 Artifact 参数版本；restore 时必须与目标 Runtime Artifact 完全一致。
    model_version: str
    # 模型 Core 状态结构版本；由 Core 定义并校验。
    state_schema_id: str
    # 最近一次成功更新使用的 feature 时间戳；初始/reset 后为 None。
    last_timestamp_ns: int | None
    # 最近一次成功提交的 Runtime update ID；初始/reset 后为 None。
    last_update_id: int | None
    # 最近一次成功更新携带的 producer frame ID；从未提供过时为 None。
    last_producer_frame_id: int | None
    # 模型特有状态数组的只读副本。
    arrays: Mapping[str, np.ndarray]
    # 模型特有的非负整数计数器；Runtime 的下一个 update ID 由 last_update_id 推导。
    counters: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_type or not self.model_version or not self.state_schema_id:
            raise ValueError("runtime-state identity fields are required")
        for name, value in (
            ("last_timestamp_ns", self.last_timestamp_ns),
            ("last_update_id", self.last_update_id),
            ("last_producer_frame_id", self.last_producer_frame_id),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when provided")
        if (self.last_timestamp_ns is None) != (self.last_update_id is None):
            raise ValueError(
                "last_timestamp_ns and last_update_id must both be set or both be None"
            )
        if self.last_update_id is None and self.last_producer_frame_id is not None:
            raise ValueError("last_producer_frame_id requires a successful Runtime update")
        if any(int(value) < 0 for value in self.counters.values()):
            raise ValueError("runtime-state counters must be non-negative")
        object.__setattr__(self, "arrays", frozen_arrays(self.arrays))
        object.__setattr__(
            self,
            "counters",
            MappingProxyType({str(key): int(value) for key, value in self.counters.items()}),
        )


@dataclass(frozen=True)
class TrainingProvenance:
    """描述一次 fit/adapt 所使用数据、代码、配置和父模型的来源信息。"""

    # 训练或自适应数据来源标识；应能定位到数据集、session 或其他稳定来源。
    source_ids: tuple[str, ...]
    # 实际参与本次拟合的逻辑样本数量。
    sample_count: int
    # 实际参与拟合的独立时序数据段数量；可对应 trial、episode、trajectory 或 segment。
    # 这是段的数量，不是 ID 列表，也不是每段的样本数。
    sequence_count: int
    # 产生参数的 Trainer/拟合代码版本，通常与 implementation_version 对齐。
    training_code_version: str
    # 训练配置经确定性 JSON 序列化后得到的哈希，用于复现和审计。
    training_config_hash: str
    # adapt 所用父 Artifact 的 model_version；根 Artifact 必须为 None。
    parent_model_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        if self.sample_count < 0 or self.sequence_count < 0:
            raise ValueError("provenance counts must be non-negative")
        if not self.training_code_version or not self.training_config_hash:
            raise ValueError("training code version and config hash are required")


@dataclass(frozen=True)
class TrainingReport:
    """SDK ``fit()`` 或 ``adapt()`` 返回的训练过程摘要。"""

    # 产生本报告的 Decoder 算法家族，必须与结果 Artifact.model_type 一致。
    model_type: str
    # 产生本报告的具体参数版本，必须与结果 Artifact.model_version 一致。
    model_version: str
    # 实际参与拟合的逻辑样本数量。
    sample_count: int
    # 实际参与拟合且不可跨边界连接的独立时序数据段数量；不是 ID 列表或每段样本数。
    sequence_count: int
    # SDK 统计的端到端训练或自适应耗时，单位为纳秒。
    duration_ns: int
    # 模型特有训练指标；所有值必须为有限浮点数，构造后冻结为只读映射。
    metrics: Mapping[str, float]
    # 面向人的非致命警告；不应作为程序控制合同。没有警告时为空元组。
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.sample_count, self.sequence_count, self.duration_ns) < 0:
            raise ValueError("training report counts and duration must be non-negative")
        object.__setattr__(self, "metrics", _finite_metrics(self.metrics))


@dataclass(frozen=True)
class EvaluationReport:
    """SDK ``evaluate()`` 对一个既有 Artifact 生成的评估摘要。"""

    # 被评估 Artifact 的 Decoder 算法家族，必须与 Artifact.model_type 一致。
    model_type: str
    # 被评估 Artifact 的具体参数版本，必须与 Artifact.model_version 一致。
    model_version: str
    # 实际纳入评估指标计算的逻辑样本数量。
    sample_count: int
    # 实际纳入评估且不可跨边界连接的独立时序数据段数量；不是 ID 列表或每段样本数。
    sequence_count: int
    # SDK 统计的端到端评估耗时，单位为纳秒。
    duration_ns: int
    # 模型特有评估指标；所有值必须为有限浮点数，构造后冻结为只读映射。
    metrics: Mapping[str, float]
    # 面向人的非致命警告；不应作为程序控制合同。没有警告时为空元组。
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.sample_count, self.sequence_count, self.duration_ns) < 0:
            raise ValueError("evaluation report counts and duration must be non-negative")
        object.__setattr__(self, "metrics", _finite_metrics(self.metrics))


@dataclass(frozen=True)
class TrainingResult:
    """一次成功 fit/adapt 返回的不可变 Artifact 与对应训练报告。"""

    # 本次拟合新生成的不可变 ModelArtifact；adapt 必须返回 child 而不能修改 parent。
    artifact: "ModelArtifact"
    # 与 artifact 的 model_type/model_version 一致的训练或自适应报告。
    report: TrainingReport
