# Use Python 3.9 image as the base
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# RUN pip install ffmpeg-python[ffmpeg]
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install ffmpeg-python

# Copy the server.py file
COPY ./server.py .

# Expose port for uvicorn
EXPOSE 8702 

# Define environment variable for OpenAI key 
ENV OPENAI_API_KEY=""

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8702"]
