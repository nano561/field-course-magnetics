# Use Python 3.10 from the respective version as a parent image
FROM python:3.10 

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the your_code_directory into the container at /app
COPY /code /app

# Copy the requirements.txt file into the container at /app/requirements.txt

COPY requirements.txt /app/requirements.txt

# These are folder outside app if there is requirement for your application
# COPY config /config
# COPY your_folder /your_folder

# For fixing ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN apt-get update
RUN apt install -y libgl1-mesa-glx

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8888

# Run app.py when the container launches
CMD ["jupyterlab"]