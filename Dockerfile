FROM tensorflow/tensorflow:2.0.0b1-gpu-py3

COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Don't install tensorflow because it'll bork the pre-installed tensorflow
RUN grep -v tensorflow requirements.txt > filtered-requirements.txt
RUN pip install -r filtered-requirements.txt

COPY ./src /app

ENTRYPOINT ["/usr/local/bin/python"]
CMD ["style_transfer.py"]
