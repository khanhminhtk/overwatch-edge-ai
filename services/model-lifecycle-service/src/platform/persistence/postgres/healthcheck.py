from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.platform.persistence.postgres.pool import PostgresPool


class PostgresHealthCheck:
    def __init__(self, pool: "PostgresPool"):
        self._pool = pool

    async def check(self) -> bool:
        try:
            async with self._pool.acquire() as connection:
                result =await connection.execute("SELECT 1")
                # print(f"[POSTGRES_HEALTH_CHECK] Result: {result}")
            return result == "SELECT 1"
        except Exception:
            return False
        

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

#     logger = Logger(name="PostgresHealthCheck")

#     postgres_pool = PostgresPool(
#         dsn=config.to_dsn(),
#         logger=logger,
#         min_size=1,
#         max_size=10,
#         command_timeout=30.0,
#     )

#     async def main():
#         await postgres_pool.start()
#         health_check = PostgresHealthCheck(pool=postgres_pool)
#         is_healthy = await health_check.check()
#         print(f"Postgres health check: {is_healthy}")
#         await postgres_pool.stop()

#     asyncio.run(main())