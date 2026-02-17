# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Force cache bust for pip install
ARG CACHEBUST=1

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY finetuned_chronos_2/ ./finetuned_chronos_2/
COPY app.py .

# Expose port
EXPOSE 8080

# Set environment variable for port
ENV PORT=8080

# Run the application
CMD ["python", "app.py"]
