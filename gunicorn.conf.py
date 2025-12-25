# gunicorn.conf.py
timeout = 300  # 5 minutes (was 30 seconds by default)
workers = 1
bind = "0.0.0.0:10000"
worker_class = "sync"
