# Gunakan image Python resmi yang stabil
FROM python:3.10-slim

# Install system dependencies yang dibutuhkan oleh OpenCV, YOLOv8, dan wget
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Buat user baru (1000) agar sesuai dengan keamanan Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Salin requirements.txt terlebih dahulu agar Docker caching berjalan efisien
COPY --chown=user backend/requirements.txt $HOME/app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade -r $HOME/app/backend/requirements.txt

# Salin semua file kode (TANPA file .pt - akan didownload di bawah)
COPY --chown=user . $HOME/app

# Download model files dari GitHub saat build
# (File .pt tidak di-push ke HF via git, melainkan didownload langsung dari GitHub)
RUN mkdir -p $HOME/app/backend/models && \
    wget -q "https://raw.githubusercontent.com/zakwanaraffi/bananaleaf-ai/5403d28fb7a67cd97ace236a4de4058988946312/backend/models/best.pt" \
         -O $HOME/app/backend/models/best.pt && \
    wget -q "https://raw.githubusercontent.com/zakwanaraffi/bananaleaf-ai/5403d28fb7a67cd97ace236a4de4058988946312/backend/models/yolov8n.pt" \
         -O $HOME/app/backend/models/yolov8n.pt && \
    echo "Models downloaded successfully" && \
    ls -lh $HOME/app/backend/models/

# Pastikan folder uploads dan saved_images ada dan bisa ditulis
RUN mkdir -p $HOME/app/backend/uploads $HOME/app/backend/saved_images

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860
ENV PORT=7860

# Jalankan Flask app melalui gunicorn di folder backend
WORKDIR $HOME/app/backend
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
