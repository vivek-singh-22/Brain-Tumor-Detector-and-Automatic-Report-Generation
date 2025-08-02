# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code and model files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get update && apt-get install -y libgl1-mesa-glx

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "brain_tumor_app.py", "--server.port=8501", "--server.enableCORS=false"]


