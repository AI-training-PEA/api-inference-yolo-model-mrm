# Build

`docker build -t yolo-cpu-api .`

# Run

`docker run -p 8000:8000 yolo-cpu-api`

# Test

`curl -X POST "http://localhost:8000/predict" -F "file=@test.jpg"`
