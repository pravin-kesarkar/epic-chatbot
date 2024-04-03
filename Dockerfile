FROM python:3.10

# Set the working directory to /app
WORKDIR /

# Copy the current directory contents into the container at /app
COPY . /

# Install any needed packages specified in requirements.txt
RUN pip install  -r requirement.txt

# Run app.py when the container launches
CMD ["python", "epic_bolster_chatbot_v1.py"]