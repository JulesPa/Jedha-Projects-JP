FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app.py app.py

EXPOSE $PORT

COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]

