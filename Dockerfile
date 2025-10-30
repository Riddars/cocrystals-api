# Этап 1: Сборка окружения
# Используем официальный образ Micromamba. Он легкий и очень быстрый.
FROM mambaorg/micromamba:1.5.8-bookworm-slim AS builder

WORKDIR /app

# Копируем файлы с зависимостями в первую очередь.
# Docker будет кэшировать этот слой, если файлы не изменятся.
COPY environment.yml requirements.txt ./

# Создаем Conda-окружение из файла environment.yml.
# Micromamba автоматически подхватит и установит pip-зависимости из requirements.txt.
RUN micromamba create -y -f environment.yml && \
    # Очищаем кэш для уменьшения размера образа
    micromamba clean -afy

# Этап 2: Создание финального образа
# Используем тот же базовый образ для финального слоя, чтобы сохранить совместимость
FROM mambaorg/micromamba:1.5.8-bookworm-slim

ARG MAMBA_ENV_NAME=test
# Указываем путь к нашему созданному окружению
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV PATH="${MAMBA_ROOT_PREFIX}/envs/${MAMBA_ENV_NAME}/bin:${PATH}"

WORKDIR /app

# Копируем созданное окружение из этапа сборки
COPY --from=builder ${MAMBA_ROOT_PREFIX} ${MAMBA_ROOT_PREFIX}

# Копируем исходный код вашего приложения
COPY . .

# Порт, на котором будет работать ваше FastAPI приложение (Uvicorn по умолчанию 8000)
EXPOSE 8000

# Команда для запуска вашего приложения
# ЗАМЕНИТЕ 'main:app' на имя вашего файла и экземпляра FastAPI
# Например, если ваш файл называется app/server.py, а приложение - 'api', команда будет "app.server:api"
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
