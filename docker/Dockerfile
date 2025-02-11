# syntax=docker/dockerfile:1
FROM python:3.9
ENV TZ=America/Montreal
#ENV DATASET CheXpert
#ENV IS_TRAINING False

# Set the working directory in the container to /app
WORKDIR /CIHMLC

# Install system dependencies required for PyOpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the host to the container
COPY ./requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code from the host to the container
COPY ./src ./src
COPY ../logs ./logs

EXPOSE 8000

CMD ["uvicorn", "src.cxr_inference_app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]