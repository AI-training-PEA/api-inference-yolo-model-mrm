# 1) Run locally

## create virtual environment

`python3 -m venv env`

## activate environment for mac, linux

`source env/bin/activate`

## activate environment for windows

`จำไม่ได้`

## install libraries

`pip install -r requirements.txt`

## run

`python main.py`

# 2) Run using Docker

## Build

`docker build -t yolo-cpu-api .`

## Run

`docker run -p 8000:8000 yolo-cpu-api`

# 3) Test

```
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@202511_DPNH_221_A2_DPNH0028_20295755_020016016498.jpg"
```
