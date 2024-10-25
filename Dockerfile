# Use the python base image
FROM python

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt from the root project folder
COPY ../requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files from the training folder
COPY . .

# Command to run your training script
CMD ["python", "training/train_model.py"]
