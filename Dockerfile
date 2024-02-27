FROM python:3.10

# Set the working directory
WORKDIR /DeepXi/

RUN apt-get update && \
    apt-get install -y python3-dev python3-venv python3-pip nano libsndfile1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
RUN python3 -m venv --system-site-packages ~/venv/DeepXi
RUN . ~/venv/DeepXi/bin/activate

# Copy the local source code into the container
COPY . /DeepXi/

RUN pip install --upgrade pip
# Install Python dependencies
RUN pip install -r requirements.txt

# Default command when container starts
# CMD cd server && python3 server.py
EXPOSE 8080
