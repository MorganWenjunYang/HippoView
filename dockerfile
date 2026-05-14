ARG PYTHON_BUILDER_IMAGE=public.ecr.aws/docker/library/python:3.12
ARG PYTHON_RUNTIME_IMAGE=public.ecr.aws/docker/library/python:3.12-slim

# Build stage - install Python dependencies.
FROM ${PYTHON_BUILDER_IMAGE} AS builder

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Runtime stage.
FROM ${PYTHON_RUNTIME_IMAGE} AS runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        libpq5 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .
COPY docker/entrypoints/*.sh /app/entrypoints/
RUN chmod +x /app/entrypoints/*.sh

EXPOSE 8000 8501

CMD ["/app/entrypoints/streamlit.sh"]
