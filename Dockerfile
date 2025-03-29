FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for GDAL/GeoPandas
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    python3-gdal \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Environment variables required for GDAL/GeoPandas
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 