from typing import TYPE_CHECKING, Any
from contextlib import asynccontextmanager

if TYPE_CHECKING:
    from src.platform.persistence.postgres.pool import PostgresPool

class PostgresTransaction:
    def __init__(self, pool: "PostgresPool"):
        self._pool = pool
        
    @asynccontextmanager
    async def transaction(self):
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                yield connection

# if __name__ == "__main__":
#     import asyncio
#     from subprocess import run
#     from src.platform.config import ConfigLoader
#     from src.platform.logger import Logger
#     from src.platform.persistence.postgres.config import PostgresConfig
#     from src.platform.persistence.postgres.pool import PostgresPool

#     pwd = run(["pwd"], capture_output=True, text=True).stdout.strip()

#     config = ConfigLoader.load(
#         PostgresConfig,
#         yaml_files=[f"{pwd}/services/model-lifecycle-service/config/model_lifecycle_orchestrator_config.yaml"],
#         env_files=[f"{pwd}/services/model-lifecycle-service/config/.env"],
#         section="PostgresSql"
#     )

#     logger = Logger(name="PostgresTransaction")

#     postgres_pool = PostgresPool(
#         dsn=config.to_dsn(),
#         logger=logger,
#         min_size=1,
#         max_size=10,
#         command_timeout=30.0,
#     )

#     async def main():
#         await postgres_pool.start()
#         transaction_manager = PostgresTransaction(pool=postgres_pool)
        
#         async with transaction_manager.transaction() as connection:
#             result = await connection.fetch("SELECT * FROM kafka_events LIMIT 5")
#             print(f"Transaction result: {result}")
        
#         await postgres_pool.stop()

#     asyncio.run(main())