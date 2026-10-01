FROM apache/airflow:3.2.1-python3.11

COPY airflow/requirements-airflow.txt /tmp/requirements-airflow.txt
USER airflow
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt
