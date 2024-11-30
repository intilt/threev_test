# Use Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY app.py app.py
COPY models/ models/
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]