FROM tensorflow/tensorflow:2.0.0b1-gpu-py3

COPY ./src /app

WORKDIR /app

ENTRYPOINT ["/usr/local/bin/python"]
CMD ["fashion_mnist_classify.py"]
