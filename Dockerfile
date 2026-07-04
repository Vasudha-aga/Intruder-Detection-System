# Official Python runtime base image
FROM python:3.10-slim

# Set environment variables for clean Python execution and Hugging Face default port
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    SPACE_ID=1

# Install Linux system dependencies required for OpenCV (libGL) and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*


# Set working directory inside container
WORKDIR /app

# Copy requirement file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create non-root user (UID 1000) for Hugging Face Spaces security compliance
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Expose port 7860 for Hugging Face Spaces web routing
EXPOSE 7860

# Command to run the application server
CMD ["python", "app.py"]
