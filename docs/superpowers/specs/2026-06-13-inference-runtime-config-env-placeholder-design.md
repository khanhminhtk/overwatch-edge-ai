# Inference Runtime Config Env Placeholder Design

## Goal

Add configuration loading for `edge/jetson/inference-runtime` so that:

- `.env` is loaded from a fixed project path.
- YAML remains the primary schema and may be selected per runtime invocation.
- YAML string values may reference environment variables with `${ENV_NAME}` syntax.
- The loader returns a fully resolved `RuntimeConfig`.

## Current State

- `RuntimeConfigLoader::load(const std::string& config_path)` reads a single YAML file.
- The loader maps YAML fields directly into `RuntimeConfig`.
- There is no `.env` loading and no placeholder interpolation.

## Chosen Approach

Keep YAML as the canonical configuration structure and add placeholder resolution during YAML parsing.

The loader will:

- Load key-value pairs from a fixed `.env` path during construction.
- Accept the YAML path through the constructor so multiple YAML files can be used without changing the `.env` source.
- Resolve any string value that exactly matches `${ENV_NAME}` before assigning it into `RuntimeConfig`.
- Prefer real process environment values first, then fall back to values loaded from the fixed `.env` file.

This preserves the current config shape while allowing secrets or deployment-specific values to stay outside YAML.

## API Design

`RuntimeConfigLoader` will move from a stateless helper to a small stateful loader:

```cpp
class RuntimeConfigLoader {
  public:
    explicit RuntimeConfigLoader(
        std::string yaml_path,
        std::string env_path = "config/.env");

    RuntimeConfig load() const;
};
```

Notes:

- `env_path` has a default fixed runtime location.
- The optional constructor override exists only to support tests or special integration scenarios.
- Callers will normally construct the loader with the YAML path only.

## Resolution Rules

For each YAML field currently mapped into `RuntimeConfig`:

- If the value is not a string placeholder, keep it unchanged.
- If the value is exactly `${NAME}`, resolve `NAME`.
- Resolution order:
  1. `std::getenv("NAME")`
  2. parsed values from the fixed `.env` file
- If neither source contains `NAME`, throw a descriptive exception.

Only exact placeholder strings are in scope for this change. Mixed strings such as `grpc://${HOST}:8001` are out of scope.

## .env Parsing Rules

The `.env` parser should support the common minimal format:

- `KEY=value`
- surrounding whitespace around key/value is trimmed
- empty lines are ignored
- lines starting with `#` are ignored

Advanced shell features such as `export KEY=value`, quoted escaping rules, or inline command substitution are out of scope for now.

## Error Handling

The loader should fail fast with clear errors for:

- missing YAML file
- malformed YAML
- empty required values such as `triton.grpc_url`
- unresolved placeholders such as `${TRITON_URL}` when the variable is not present
- malformed placeholder syntax if the string starts with `${` but does not end cleanly with `}`

## Implementation Shape

Expected code changes:

- update `src/infra/runtime_config_loader.hpp`
  - store YAML path, env path, and parsed env map
  - add helper methods for env parsing and placeholder resolution
- update `src/infra/runtime_config_loader.cpp`
  - parse `.env`
  - resolve YAML string values before assignment
- update `src/main.cpp`
  - construct `RuntimeConfigLoader` with the YAML path
  - call `load()` with no parameter
- update `config/runtime.example.yaml`
  - document placeholder usage with `${ENV_NAME}` examples

## Testing

Add focused tests around loader behavior:

- loads plain YAML values unchanged
- resolves `${ENV_NAME}` from process environment
- falls back to fixed `.env` values when process environment is absent
- throws on missing placeholder variables

If there is no existing unit test target, add a minimal loader-focused test target rather than broad integration coverage.

## Non-Goals

- recursive resolution across arbitrary YAML trees
- partial string interpolation
- writing values back into YAML
- hot reload of config files

## Open Decisions Already Resolved

- `.env` path is fixed by default: yes
- YAML path is provided when constructing the loader: yes
- default `.env` path may still be overridden in tests: yes
