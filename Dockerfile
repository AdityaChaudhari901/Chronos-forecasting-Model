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

# Install Python dependencies with timestamp to force rebuild
# This bypasses Kaniko's aggressive caching
RUN echo "Build timestamp: 2026-02-17-15:11" && \
    pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY finetuned_chronos_2/ ./finetuned_chronos_2/
COPY app.py .

# Expose port
EXPOSE 8080

# Set environment variable for port
ENV PORT=8080

# Run the application
CMD ["python", "app.py"]
