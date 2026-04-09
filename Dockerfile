FROM python:3.9

WORKDIR /app

# ✅ Install system dependencies (THIS FIXES YOUR ERROR)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "train.py"]