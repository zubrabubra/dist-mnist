FROM tensorflow/tensorflow:1.1.0-rc1

ENV DATA_DIR=/var/tensorflow/

RUN apt-get update
RUN apt-get install wget
RUN wget https://github.com/stedolan/jq/releases/download/jq-1.5/jq-linux64
RUN chmod +x jq-linux64
RUN mv jq-linux64 /usr/bin/jq

COPY mnist.py /mnist.py
COPY run.sh /run.sh

RUN chmod +x /run.sh

CMD ["/run.sh"]
