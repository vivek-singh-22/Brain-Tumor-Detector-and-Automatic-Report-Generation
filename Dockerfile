# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install required system packages
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Hugging Face CLI for pulling model if needed dynamically
RUN pip install huggingface_hub

# Expose Streamlit port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "brain_tumor_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
