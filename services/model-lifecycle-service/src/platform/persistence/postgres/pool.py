from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from src.platform.logger import Logger
from src.platform.persistence.postgres.errors import PostgresNotStartedError

class PostgresPool:
    def __init__(
        self,
        dsn: str,
        logger: Logger,
        *,
        min_size: int = 1,
        max_size: int = 10,
        command_timeout: float = 30.0,
    ):
        self._dsn = dsn
        self._logger = logger
        self._min_size = min_size
        self._max_size = max_size
        self._command_timeout = command_timeout
        self._pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        if self._pool is not None:
            self._logger.warning(
                "[POSTGRES_POOL_ALREADY_STARTED] Postgres pool is already started"
            )

        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                command_timeout=self._command_timeout,
            )
            self._logger.info(
                "[POSTGRES_POOL_STARTED] Postgres pool started successfully"
            )

    async def stop(self) -> None:
        if self._pool is None:
            self._logger.warning(
                "[POSTGRES_POOL_NOT_STARTED] Postgres pool is not started"
            )
            return

        await self._pool.close()
        self._pool = None
        self._logger.info(
            "[POSTGRES_POOL_STOPPED] Postgres pool stopped successfully"
        )

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._logger.error(
                "[POSTGRES_POOL_NOT_STARTED] Postgres pool is not started"
            )
            raise PostgresNotStartedError("Postgres pool is not started")
        return self._pool
    
    @asynccontextmanager
    async def acquire(self) -> Any:
        if self._pool is None:
            self._logger.error(
                "[POSTGRES_POOL_NOT_STARTED] Postgres pool is not started"
            )
            raise PostgresNotStartedError("Postgres pool is not started")

        async with self._pool.acquire() as connection:
            yield connection


# if __name__ == "__main__":
#     from subprocess import run
#     import asyncio
#     from src.platform.config import ConfigLoader
#     from src.platform.logger import Logger
#     from src.platform.persistence.postgres.config import PostgresConfig

#     pwd = run(["pwd"], capture_output=True, text=True).stdout.strip()

#     config = ConfigLoader.load(
#         PostgresConfig,
#         yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
#         env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
#         section="PostgresSql"
#     )

#     logger = Logger(name="PostgresPool")

#     postgres_pool = PostgresPool(
#         dsn=config.to_dsn(),
#         logger=logger,
#         min_size=1,
#         max_size=10,
#         command_timeout=30.0,
#     )

#     async def main():
#         await postgres_pool.start()
#         pool = postgres_pool.pool
#         async with pool.acquire() as connection:
#                 result = await connection.fetch("SELECT 1")
#                 print(result)

#     asyncio.run(main())