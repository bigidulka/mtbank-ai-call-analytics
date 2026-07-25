"""Строгая загрузка versioned policy packs без network или dynamic code."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from pydantic import Field, ValidationError, field_validator, model_validator

from mtbank_ai.domain.agents import ComplianceSeverity
from mtbank_ai.domain.base import Confidence, LongText, NonEmptyId, StrictFrozenModel

_POLICY_COMPONENT = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_POLICY_NAMES = ("taxonomy", "quality", "compliance")
PolicyName: TypeAlias = Literal["taxonomy", "quality", "compliance"]
_TOPIC_IDS = frozenset({"кредиты", "карты", "переводы", "жалобы", "другое"})
_PRIORITY_IDS = frozenset({"low", "medium", "high"})
_QUALITY_CRITERIA = ("greeting", "need_detection", "solution_provided", "farewell")


class PolicyLoadError(ValueError):
    """Policy pack не прошёл containment, syntax или semantic validation."""


class PolicyMetadata(StrictFrozenModel):
    policy_id: NonEmptyId
    version: NonEmptyId
    owner: NonEmptyId
    effective_date: NonEmptyId

    @field_validator("effective_date")
    @classmethod
    def require_iso_date(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("effective_date должен быть ISO-8601 датой") from error
        if parsed.isoformat() != value:
            raise ValueError("effective_date должен быть canonical ISO-8601 датой")
        return value


class TaxonomyTopic(StrictFrozenModel):
    id: NonEmptyId
    description: LongText
    allowed_priorities: tuple[NonEmptyId, ...] = Field(min_length=1)

    @field_validator("allowed_priorities", mode="before")
    @classmethod
    def parse_priorities(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_topic(self) -> TaxonomyTopic:
        if self.id not in _TOPIC_IDS:
            raise ValueError("topic не входит в утверждённую таксономию")
        if len(set(self.allowed_priorities)) != len(self.allowed_priorities):
            raise ValueError("allowed_priorities должны быть уникальны")
        if not set(self.allowed_priorities).issubset(_PRIORITY_IDS):
            raise ValueError("priority не входит в утверждённую таксономию")
        return self


class TaxonomyPolicy(StrictFrozenModel):
    metadata: PolicyMetadata
    topics: tuple[TaxonomyTopic, ...] = Field(min_length=1)

    @field_validator("topics", mode="before")
    @classmethod
    def parse_topics(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_topics(self) -> TaxonomyPolicy:
        identifiers = tuple(topic.id for topic in self.topics)
        if len(set(identifiers)) != len(identifiers) or set(identifiers) != _TOPIC_IDS:
            raise ValueError("taxonomy должна содержать ровно утверждённые topics")
        return self

    def allowed_priorities(self, topic: str) -> tuple[str, ...]:
        for item in self.topics:
            if item.id == topic:
                return item.allowed_priorities
        raise PolicyLoadError("topic отсутствует в taxonomy")


class QualityCriterion(StrictFrozenModel):
    id: NonEmptyId
    weight: float = Field(gt=0.0, le=1.0)
    description: LongText


class QualityPolicy(StrictFrozenModel):
    metadata: PolicyMetadata
    criteria: tuple[QualityCriterion, ...] = Field(min_length=1)
    review_confidence_threshold: Confidence
    role_confidence_threshold: Confidence

    @field_validator("criteria", mode="before")
    @classmethod
    def parse_criteria(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_rubric(self) -> QualityPolicy:
        identifiers = tuple(item.id for item in self.criteria)
        if identifiers != _QUALITY_CRITERIA:
            raise ValueError("quality criteria должны иметь утверждённый порядок и состав")
        total = sum((Decimal(str(item.weight)) for item in self.criteria), Decimal("0"))
        if total != Decimal("1"):
            raise ValueError("quality weights должны суммироваться ровно до 1")
        return self

    def criterion(self, identifier: str) -> QualityCriterion:
        for item in self.criteria:
            if item.id == identifier:
                return item
        raise PolicyLoadError("criterion отсутствует в quality rubric")


class ComplianceRule(StrictFrozenModel):
    id: NonEmptyId
    severity: ComplianceSeverity
    description: LongText

    @field_validator("severity", mode="before")
    @classmethod
    def parse_severity(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return ComplianceSeverity(value)
            except ValueError as error:
                raise ValueError("severity не поддерживается") from error
        return value


class CompliancePolicy(StrictFrozenModel):
    metadata: PolicyMetadata
    rules: tuple[ComplianceRule, ...] = Field(min_length=1)

    @field_validator("rules", mode="before")
    @classmethod
    def parse_rules(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_rules(self) -> CompliancePolicy:
        identifiers = tuple(rule.id for rule in self.rules)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("compliance rule IDs должны быть уникальны")
        return self

    def rule(self, identifier: str) -> ComplianceRule:
        for item in self.rules:
            if item.id == identifier:
                return item
        raise PolicyLoadError("rule отсутствует в compliance policy")


Policy = TaxonomyPolicy | QualityPolicy | CompliancePolicy
PolicyType = TypeVar("PolicyType", bound=Policy)


@dataclass(frozen=True, slots=True)
class LoadedPolicyPack(Generic[PolicyType]):
    name: PolicyName
    version: str
    owner: str
    effective_date: str
    sha256: str
    policy: PolicyType


_POLICY_MODELS: dict[str, type[Policy]] = {
    "taxonomy": TaxonomyPolicy,
    "quality": QualityPolicy,
    "compliance": CompliancePolicy,
}


def load_policy_pack(
    name: PolicyName,
    version: str = "v1",
    *,
    root: Path | None = None,
) -> LoadedPolicyPack[Policy]:
    """Загружает только reviewed JSON-compatible YAML под package root.

    JSON является строгим подмножеством YAML; это исключает неявные YAML tags,
    aliases и type coercion в критичной policy boundary.
    """

    _validate_component(name)
    _validate_component(version)
    if name not in _POLICY_NAMES:
        raise PolicyLoadError("неизвестный policy pack")
    policy_root = _resolve_root(root or Path(__file__).resolve().parent)
    path = _resolve_policy_path(policy_root, name, version)
    try:
        raw = path.read_bytes()
        decoded = raw.decode("utf-8")
        document = json.loads(decoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PolicyLoadError("policy pack должен быть валидным UTF-8 JSON-compatible YAML") from error
    if not isinstance(document, dict):
        raise PolicyLoadError("policy pack должен быть JSON object")

    model = _POLICY_MODELS[name]
    try:
        policy = model.model_validate(document, strict=True)
    except ValidationError as error:
        raise PolicyLoadError("policy pack не соответствует строгой schema") from error
    if policy.metadata.policy_id != name or policy.metadata.version != version:
        raise PolicyLoadError("policy metadata не совпадает с path")

    return LoadedPolicyPack(
        name=cast(PolicyName, name),
        version=policy.metadata.version,
        owner=policy.metadata.owner,
        effective_date=policy.metadata.effective_date,
        sha256=hashlib.sha256(raw).hexdigest(),
        policy=policy,
    )


class PolicyRegistry:
    """Инициализируется один раз и удерживает immutable verified packs."""

    def __init__(self, root: Path | None = None, *, version: str = "v1") -> None:
        self._root = root or Path(__file__).resolve().parent
        self._version = version
        self._taxonomy: LoadedPolicyPack[TaxonomyPolicy] | None = None
        self._quality: LoadedPolicyPack[QualityPolicy] | None = None
        self._compliance: LoadedPolicyPack[CompliancePolicy] | None = None

    @property
    def taxonomy(self) -> LoadedPolicyPack[TaxonomyPolicy]:
        if self._taxonomy is None:
            loaded = load_policy_pack("taxonomy", self._version, root=self._root)
            self._taxonomy = LoadedPolicyPack(
                name="taxonomy",
                version=loaded.version,
                owner=loaded.owner,
                effective_date=loaded.effective_date,
                sha256=loaded.sha256,
                policy=cast(TaxonomyPolicy, loaded.policy),
            )
        return self._taxonomy

    @property
    def quality(self) -> LoadedPolicyPack[QualityPolicy]:
        if self._quality is None:
            loaded = load_policy_pack("quality", self._version, root=self._root)
            self._quality = LoadedPolicyPack(
                name="quality",
                version=loaded.version,
                owner=loaded.owner,
                effective_date=loaded.effective_date,
                sha256=loaded.sha256,
                policy=cast(QualityPolicy, loaded.policy),
            )
        return self._quality

    @property
    def compliance(self) -> LoadedPolicyPack[CompliancePolicy]:
        if self._compliance is None:
            loaded = load_policy_pack("compliance", self._version, root=self._root)
            self._compliance = LoadedPolicyPack(
                name="compliance",
                version=loaded.version,
                owner=loaded.owner,
                effective_date=loaded.effective_date,
                sha256=loaded.sha256,
                policy=cast(CompliancePolicy, loaded.policy),
            )
        return self._compliance

    def load_all(self) -> tuple[LoadedPolicyPack[Policy], ...]:
        return (
            cast(LoadedPolicyPack[Policy], self.taxonomy),
            cast(LoadedPolicyPack[Policy], self.quality),
            cast(LoadedPolicyPack[Policy], self.compliance),
        )


def _validate_component(value: str) -> None:
    if not isinstance(value, str) or not _POLICY_COMPONENT.fullmatch(value):
        raise PolicyLoadError("policy name и version должны быть безопасными простыми компонентами")


def _resolve_root(root: Path) -> Path:
    if root.is_symlink():
        raise PolicyLoadError("policy root не может быть symlink")
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise PolicyLoadError("policy root недоступен") from error
    if not resolved.is_dir():
        raise PolicyLoadError("policy root должен быть каталогом")
    return resolved


def _resolve_policy_path(root: Path, name: str, version: str) -> Path:
    candidate = root / name / f"{version}.yaml"
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise PolicyLoadError("policy path выходит за root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PolicyLoadError("symlink в policy path запрещён")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PolicyLoadError("policy pack отсутствует") from error
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise PolicyLoadError("policy path выходит за root")
    return resolved
