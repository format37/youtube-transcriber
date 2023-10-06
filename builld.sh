#!/bin/bash
# sudo docker build -t youtube-transcriber .

# Prompt for API key
# read -p "Enter OpenAI API key: " OPENAI_API_KEY  

# Set as environment variable
# export OPENAI_API_KEY=$OPENAI_API_KEY  

# Build image
docker-compose build

# Start container and pass API key
# docker-compose up -d -e OPENAI_API_KEY=$OPENAI_API_KEY

# docker-compose logs -f