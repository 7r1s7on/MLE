docker build --target trainer -t tabnet-trainer .

docker build --target inference -t tabnet-inference .

docker run -it --rm   -v "$(pwd)/data:/app/data"   -v "$(pwd)/logs:/app/logs"   -v "$(pwd)/models:/app/models"   tabnet-trainer

docker run -it --rm   -v "$(pwd)/data:/app/data"   -v "$(pwd)/logs:/app/logs"   -v "$(pwd)/models:/app/models" -v "$(pwd)/output:/app/output"  tabnet-inference
