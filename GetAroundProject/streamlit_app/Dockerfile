FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY get_around_delay_analysis.xlsx /get_around_delay_analysis.xlsx

COPY app/ app/
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE $PORT

CMD ["/start.sh"]