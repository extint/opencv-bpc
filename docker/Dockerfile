FROM vaheta/pose-estimator-base:latest
RUN apt-get update
RUN apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libx11-dev libxext-dev libxrender-dev python3-pip
RUN python3 -m pip install --upgrade setuptools --break-system-packages
COPY requirements.txt .
RUN pip install -r requirements.txt --break-system-packages
