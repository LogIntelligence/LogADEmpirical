FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt update
RUN apt install -y build-essential

COPY requirements.txt .
RUN pip install --ignore-installed -r requirements.txt
RUN pip install transformers

COPY . .
RUN pip install --upgrade numpy
RUN python ./main_run.py --help

CMD /bin/bash
