FROM ristar20/u20:pytorch1.7.1-cuda11.0

# Build-------------
# docker build --build-arg UID=`id -u` GID=`id -g` local:wslocalization .

# Devel-------------
# docker run -it --gpus all --rm -e DISPLAY -e XAUTHORITY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix --name "tros" -v /mnt/Data/docker_v/do:/home/do local:wslocalization

# Run --------------
# docker run -it --gpus all --rm ---name "tros" -v /mnt/Data/docker_v/do:/home/do local:wslocalization python3 streaming_detection.py --cuda --resume ./ckpt/2020-12-07_20-40-44_detection/model/checkpoint-epoch19.pth

ARG UID=1000
ARG GID=1000

RUN  usermod  --uid $UID $USER_NAME && \
  groupmod --gid $GID $USER_NAME

COPY requirements.txt    /apps/scripts
RUN pip3 install -r /apps/scripts/requirements.txt
