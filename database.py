import psycopg
from psycopg.rows import dict_row


def initialize_database():
    # libpq reads PGHOST, PGDATABASE, PGUSER and PGPASSWORD from Compose.
    with psycopg.connect(connect_timeout=5) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                created_at timestamptz NOT NULL DEFAULT now(),
                article_preview varchar(200) NOT NULL,
                label text NOT NULL CHECK (label IN ('FAKE', 'REAL')),
                confidence double precision NOT NULL CHECK (confidence BETWEEN 0 AND 100)
            )
        """)


def save_prediction(article, label, confidence):
    with psycopg.connect(connect_timeout=5) as connection:
        connection.execute(
            "INSERT INTO predictions (article_preview, label, confidence) VALUES (%s, %s, %s)",
            (article.strip()[:200], label, confidence),
        )


def recent_predictions(limit=10):
    with psycopg.connect(connect_timeout=5, row_factory=dict_row) as connection:
        return connection.execute(
            """
            SELECT created_at, article_preview, label, confidence
            FROM predictions ORDER BY id DESC LIMIT %s
            """,
            (limit,),
        ).fetchall()


if __name__ == "__main__":
    initialize_database()
