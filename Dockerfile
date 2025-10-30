FROM mambaorg/micromamba:1.5.8-bookworm-slim AS builder
WORKDIR /app

COPY environment.yml pyproject.toml ./
RUN micromamba create -y -f environment.yml && \
    micromamba run -n test /bin/bash -c "uv pip install --no-cache ." && \
    micromamba clean -afy

FROM mambaorg/micromamba:1.5.8-bookworm-slim
ARG MAMBA_ENV_NAME=test
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV PATH="${MAMBA_ROOT_PREFIX}/envs/${MAMBA_ENV_NAME}/bin:${PATH}"
WORKDIR /app
COPY --from=builder ${MAMBA_ROOT_PREFIX} ${MAMBA_ROOT_PREFIX}
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]

#FROM python:3.8
#WORKDIR /app
#COPY ./requirements.txt .
##RUN apt-get update && apt-get install openbabel -y
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    libopenbabel-dev
#
#RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#
#RUN pip install -r requirements.txt --no-cache
#
#COPY . .
#EXPOSE 8000
#CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0"]
#
