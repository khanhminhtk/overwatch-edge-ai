from dataclasses import dataclass

@dataclass(frozen=True)
class PostgrestConfig:
    user: str
    password: str
    database: str
    host: str
    port: int

