FROM python:3.7.3

RUN mkdir -p /app
copy scanner /app

RUN pip install pandas
RUN pip install boto3==1.10.28
RUN pip install botocore==1.13.28
RUN pip install certifi==2019.11.28
RUN pip install chardet==3.0.4
RUN pip install Cython==0.29.14
RUN pip install docutils==0.15.2
RUN pip install grpcio==1.25.0
RUN pip install grpcio-tools==1.25.0
RUN pip install idna==2.8
RUN pip install imutils==0.5.3
RUN pip install jmespath==0.9.4
RUN pip install numpy==1.17.4
RUN pip install opencv-python==4.1.2.30
RUN pip install pandas==0.25.3
RUN pip install Pillow==6.2.1
RUN pip install protobuf==3.11.1
RUN pip install pytesseract==0.3.0
RUN pip install python-dateutil==2.8.0
RUN pip install pytz==2019.3
RUN pip install requests==2.22.0
RUN pip install s3transfer==0.2.1
RUN pip install six==1.13.0
RUN pip install tesseract==0.1.3
RUN pip install urllib3==1.25.7
RUN apt-get update
RUN apt-get install vim -y
#RUN export PYTHONPATH='$PYTHONPATH:/app/scanner/'
EXPOSE 50051

cmd sleep 10000000000000000000000 
