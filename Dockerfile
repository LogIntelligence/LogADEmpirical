FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# Update system
RUN apt update
RUN apt install -y build-essential

# Install Requirements
COPY requirements.txt .
RUN pip install --ignore-installed -r requirements.txt
RUN pip install transformers
RUN pip install --upgrade numpy

# Copy over files and run
COPY . .
RUN python ./main_run.py --help

# Enter and run what you like
CMD /bin/bash
