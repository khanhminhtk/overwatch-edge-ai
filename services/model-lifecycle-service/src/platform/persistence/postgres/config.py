from pydantic.dataclasses import dataclass

@dataclass(
    frozen=True,
    kw_only=True,
    slots=True,
)
class PostgresConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("Postgres host must not be empty")

        if self.port <= 0 or self.port > 65535:
            raise ValueError(
                f"Postgres port must be in the range 1-65535: {self.port}"
            )

        if not self.database.strip():
            raise ValueError("Postgres database must not be empty")

        if not self.user.strip():
            raise ValueError("Postgres user must not be empty")

        if not self.password.strip():
            raise ValueError("Postgres password must not be empty")
        
    def to_dict(self) -> dict[str, str | int]:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }
    
    def to_dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )
    
# if __name__ == "__main__":
#     from src.platform.config import ConfigLoader
#     from subprocess import run

#     pwd = run(["pwd"], capture_output=True, text=True).stdout.strip()

#     config = ConfigLoader.load(
#         PostgresConfig,
#         yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
#         env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
#         section="PostgresSql"
#     )

#     print(config)
#     print(f"DSN: {config.to_dsn()}")
#     print(f"Dict: {config.to_dict()}")