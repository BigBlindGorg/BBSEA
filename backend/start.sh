#!/bin/bash
cd /var/www/bbsea/backend
source venv/bin/activate
exec gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001 --timeout 300
