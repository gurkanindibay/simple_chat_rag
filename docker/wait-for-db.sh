#!/usr/bin/env bash
set -e

# usage: wait-for-db.sh <cmd...>
echo "Waiting for database..."
while true; do
    python - <<'PY'
import os, sys
from urllib.parse import urlparse
try:
        import psycopg
except Exception:
        # psycopg might not be installed yet; signal failure
        sys.exit(1)

url = os.environ.get('DATABASE_URL')
if not url:
        print('DATABASE_URL not set')
        sys.exit(2)
u = urlparse(url)
host = u.hostname
port = u.port or 5432
user = u.username
pwd = u.password
dbname = u.path.lstrip('/')
try:
        conn = psycopg.connect(host=host, port=port, user=user, password=pwd, dbname=dbname, connect_timeout=2)
        conn.close()
        print('Database is up')
        sys.exit(0)
except Exception:
        sys.exit(1)
PY

    if [ $? -eq 0 ]; then
        break
    fi
    echo "DB not ready yet. Sleeping 1s..."
    sleep 1
done

exec "$@"
