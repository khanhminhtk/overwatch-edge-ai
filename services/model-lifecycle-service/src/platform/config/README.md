# `config_loader_v2`

Drop-in typed config loader package. Copy the `config_loader/` directory into another
project's `utils/` directory, then import `ConfigLoader` from there.

## What It Does

- merges multiple YAML files in order
- merges multiple `.env` files in order
- optionally overlays runtime environment values
- resolves `${VAR}` and `${VAR:-default}` placeholders
- maps resolved config into nested dataclasses at any depth
- supports `Enum`, `Path`, `Optional`, `Union`, `list`, `dict`, and `tuple`

## Drop-in Package Layout

```text
your_project/
└── utils/
    └── config_loader/
        ├── __init__.py
        ├── facade.py
        ├── mapper.py
        ├── provider.py
        ├── sources.py
        ├── types.py
        └── py.typed       # PEP 561 type checker marker
```

## Installation Style

Copy the `config_loader/` directory into your target project's `utils/` package.

Required external dependencies:

- `PyYAML`
- `python-dotenv`

## Quick Usage

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from utils.config_loader import ConfigLoader


class Mode(Enum):
    DEV = "dev"
    PROD = "prod"


@dataclass(frozen=True)
class TlsConfig:
    enabled: bool
    cert_path: Path


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    mode: Mode
    tls: TlsConfig


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig
    tags: list[str]


config = ConfigLoader.load(
    AppConfig,
    yaml_files=[
        "config/base.yaml",
        "config/override.yaml",
    ],
    env_files=[
        ".env",
        ".env.local",
    ],
    env={
        "API_HOST": "runtime.local",
        "APP_MODE": "prod",
        "TLS_CERT_PATH": "/run/secrets/prod.pem",
    },
)

print(config.api.host)
print(config.api.port)
print(config.api.tls.cert_path)
```

## Nested Mapping at Any Depth

The mapper recursively traverses the YAML structure and the dataclass type tree.
If your YAML has N levels of nesting, define N matching dataclass types and the
mapper will handle the rest automatically.

```python
@dataclass(frozen=True)
class TopicConfig:
    name: str
    partitions: int

@dataclass(frozen=True)
class KafkaConfig:
    enabled: bool
    topics: list[TopicConfig]

@dataclass(frozen=True)
class SinkConfig:
    kafka: KafkaConfig

@dataclass(frozen=True)
class Level3Config:
    sink: SinkConfig

@dataclass(frozen=True)
class Level2Config:
    level3: Level3Config

@dataclass(frozen=True)
class Level1Config:
    level2: Level2Config

# YAML with 5 levels → dataclass auto-maps
# level1.level2.level3.sink.kafka.topics[0].name
```

Supported types in nested fields:

| Python type | YAML equivalent |
|---|---|
| `str` | string |
| `int` | integer |
| `float` | float |
| `bool` | boolean, or string `"true"`/`"false"`/`"yes"`/`"no"`/`"on"`/`"off"` |
| `Path` | string (converted to `pathlib.Path`) |
| `Enum` | string (matched by value) |
| `Optional[T]` | value or `null` |
| `Union[T, U]` | value (first matching type wins) |
| `list[T]` | array |
| `dict[K, V]` | mapping |
| `tuple[T, ...]` | array |
| dataclass | nested mapping |
| `Any` | any value (passthrough) |

## Validation with `__post_init__`

Since the mapper constructs dataclasses via `target_type(**mapped_values)`, any
`__post_init__` method on your dataclass is automatically called. Use it for
business validation:

```python
@dataclass(frozen=True)
class AppConfig:
    port: int
    host: str

    def __post_init__(self):
        if not 1024 <= self.port <= 65535:
            raise ValueError(f"port must be 1024-65535, got {self.port}")
        if not self.host.strip():
            raise ValueError("host must not be empty")
```

## Extra YAML Keys

Keys in the YAML data that do not match any field in the target dataclass are
silently ignored.

## Public API

- `ConfigLoader.load(MyDataclass, yaml_files=None, env_files=None, env=None, include_os_env=True, section=None)`
- `ConfigProvider.get_raw(section=None)` for advanced/manual composition
- `ConfigProvider.require(MyDataclass, section=None)` for advanced/manual composition
- `YamlConfigSource(paths)`
- `DotenvEnvSource(paths)`
- `EnvironmentEnvSource(values=None)`

## Example Files

Reusable sample files and a demo runner:

- `examples/base.yaml`
- `examples/override.yaml`
- `examples/env_app.yaml`
- `examples/base.env`
- `examples/override.env`
- `examples/section.yaml`
- `examples/run_demo.py` — run via `python examples/run_demo.py`

## Tests

```bash
# from this directory
pip install -e .
pytest tests/ -v
```
