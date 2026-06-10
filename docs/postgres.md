# PostgreSQL Migration Guide

## Why migrate from SQLite

SQLite with WAL works fine for a single-user deployment but hits contention under concurrent API requests and can't be accessed by multiple processes (e.g. a separate analytics worker). Migrate to Postgres before going multi-user or containerising the API behind a load balancer.

## 1. Provision a Postgres instance

```bash
# Docker (local dev / staging)
docker run -d \
  --name f1di-postgres \
  -e POSTGRES_USER=f1di \
  -e POSTGRES_PASSWORD=change_me \
  -e POSTGRES_DB=f1di \
  -p 5432:5432 \
  postgres:16-alpine

# Or with docker compose — add to docker-compose.yml:
#   postgres:
#     image: postgres:16-alpine
#     environment:
#       POSTGRES_USER: f1di
#       POSTGRES_PASSWORD: change_me
#       POSTGRES_DB: f1di
#     ports: ["5432:5432"]
#     volumes: ["pgdata:/var/lib/postgresql/data"]
```

## 2. Install the Postgres driver

```bash
uv add psycopg2-binary
# or: pip install psycopg2-binary
```

## 3. Update .env

```bash
F1DI_STORAGE_URL=postgresql://f1di:change_me@localhost:5432/f1di
```

Remove (or keep) `F1DI_STORAGE_URL=sqlite:///./f1di.db` — the new value takes precedence.

## 4. Run Alembic migrations

The migrations in `migrations/` were generated from the SQLAlchemy models and are dialect-agnostic.

```bash
# Point Alembic at your Postgres URL
export DATABASE_URL=postgresql://f1di:change_me@localhost:5432/f1di

# Create tables (first-time setup)
alembic upgrade head

# Or: let the API do it automatically on startup —
# Base.metadata.create_all() runs in the lifespan and is idempotent.
```

## 5. Migrate existing data from SQLite (optional)

If you have insights or feedback in `f1di.db` worth keeping:

```bash
# Install pgloader
sudo apt install pgloader   # Debian/Ubuntu

# Migrate
pgloader \
  sqlite:///./f1di.db \
  postgresql://f1di:change_me@localhost:5432/f1di
```

`pgloader` handles type coercion (SQLite TEXT timestamps → Postgres TIMESTAMP) automatically.

## 6. Verify

```bash
curl http://localhost:8080/ready
# "database": true  ← confirms the new connection works
```

## Connection pool settings

The API automatically uses `pool_size=5, max_overflow=10, pool_pre_ping=True` for non-SQLite URLs (see `src/f1di/storage/database.py`). Adjust via a custom `F1DI_STORAGE_URL` with pgbouncer or PgCat in front for high-concurrency deployments.

## Production checklist

- [ ] `F1DI_ENV=production` — enables runtime validation of required env vars
- [ ] `F1DI_STORAGE_URL` points to Postgres (not SQLite)
- [ ] `F1DI_VECTOR_BACKEND=qdrant` (memory backend is blocked in production)
- [ ] SSL: append `?sslmode=require` to the URL for managed Postgres (e.g. RDS, Cloud SQL)
- [ ] Backups: enable continuous WAL archiving or daily `pg_dump` snapshots
