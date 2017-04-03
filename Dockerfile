FROM tensorflow/tensorflow:1.0.1

ENV DATA_DIR=/var/tensorflow/

COPY mnist.py /mnist.py
COPY run.sh /run.sh

RUN chmod +x /run.sh

CMD ["/run.sh"]