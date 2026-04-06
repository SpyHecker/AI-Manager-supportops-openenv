# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port (HF uses 7860)
EXPOSE 7860

# Run server
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]