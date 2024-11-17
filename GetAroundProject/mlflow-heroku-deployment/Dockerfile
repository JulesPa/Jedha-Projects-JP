FROM continuumio/miniconda3


WORKDIR /home/app


RUN apt-get update && \
    apt-get install -y nano unzip curl && \
    curl -fsSL https://get.deta.dev/cli.sh | sh && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws


COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt


COPY app /app
COPY get_around_pricing_project.csv /get_around_pricing_project.csv


EXPOSE $PORT 8501


COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]