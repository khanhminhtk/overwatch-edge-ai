class PostgresError(RuntimeError):
    pass


class PostgresNotStartedError(PostgresError):
    pass


class PostgresConnectionError(PostgresError):
    pass


class PostgresTransactionError(PostgresError):
    pass