release: python manage.py migrate --noinput
web: gunicorn core.wsgi --log-file - --bind 0.0.0.0:$PORT
