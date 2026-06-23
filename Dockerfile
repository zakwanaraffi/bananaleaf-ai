# Gunakan image Python resmi yang stabil
FROM python:3.10-slim

# Install system dependencies yang dibutuhkan oleh OpenCV dan YOLOv8
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Buat user baru (1000) agar sesuai dengan keamanan Hugging Face Spaces (tidak boleh root)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory ke home directory user
WORKDIR $HOME/app

# Salin requirements.txt terlebih dahulu agar Docker caching berjalan efisien
COPY --chown=user backend/requirements.txt $HOME/app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade -r $HOME/app/backend/requirements.txt

# Salin semua file proyek ke container
COPY --chown=user . $HOME/app

# Pastikan folder uploads dan saved_images ada dan bisa ditulis oleh user 1000
RUN mkdir -p $HOME/app/backend/uploads $HOME/app/backend/saved_images

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Set environment variable PORT ke 7860
ENV PORT=7860

# Jalankan Flask app melalui gunicorn di folder backend
WORKDIR $HOME/app/backend
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
