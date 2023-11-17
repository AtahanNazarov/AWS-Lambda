FROM karatayevm/dino-dragon-lambda:latest

COPY lite.py .
COPY requirements.txt .

RUN pip install tflite_runtime==2.7.0

RUN pip install -r ./requirements.txt

CMD ["lite.lambda_handler"]



