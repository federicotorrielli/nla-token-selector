"""pydantic-settings configuration tree.

One `Settings(BaseSettings)` tree is constructed at the top of each
`experiments/run_*.py` and threaded down. No global state.

Sources, in precedence order (later loses to earlier on key collision):

    init kwargs  >  env vars  >  YAML (overlay > base)  >  .env  >  secrets

YAML files are passed via `yaml_file=[...]` — the per-experiment overlay
is picked from env var `CNLA_EXPERIMENT_FILE`, falling back to
`configs/experiments/default.yaml` so a bare `Settings()` resolves
during smoke tests.

Env-var overrides use `CNLA_` prefix + `__` nested delimiter, e.g.

    CNLA_EXPERIMENT__MODEL_SHORT=q7
    CNLA_MODELS__Q7__SGLANG_URL=http://node-42:30000

See `docs/pydantic_settings.md` for the captured API patterns this file
relies on.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_YAML = _REPO_ROOT / "configs" / "base.yaml"
_DEFAULT_OVERLAY = _REPO_ROOT / "configs" / "experiments" / "default.yaml"


def _yaml_files() -> list[Path]:
    overlay = Path(os.environ.get("CNLA_EXPERIMENT_FILE", _DEFAULT_OVERLAY))
    return [_BASE_YAML, overlay]


class JudgeSettings(BaseModel):
    primary_url: str
    primary_model_id: str
    secondary_url: str | None = None
    secondary_model_id: str | None = None


class ModelSettings(BaseModel):
    short: str
    base_model: str
    av_repo: str
    ar_repo: str
    layer: int
    d_model: int
    sglang_url: str
    judge: JudgeSettings


class ExperimentSettings(BaseModel):
    model_short: str
    n_positions: int = 6000
    msa_k: int = 20
    sts_window: int = 10
    temperature: float = 1.0
    code_version: int = 1


class PathsSettings(BaseModel):
    results_root: Path
    corpus_parquet: Path

    @field_validator("results_root", "corpus_parquet", mode="before")
    @classmethod
    def _expanduser(cls, v: str | Path) -> Path:
        return Path(v).expanduser()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CNLA_",
        env_nested_delimiter="__",
        yaml_file=_yaml_files(),
        extra="ignore",
    )

    paths: PathsSettings
    models: dict[str, ModelSettings]
    experiment: ExperimentSettings

    @property
    def model(self) -> ModelSettings:
        """Resolve the active model from the registry by `experiment.model_short`."""
        return self.models[self.experiment.model_short]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
