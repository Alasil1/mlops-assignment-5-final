# Use python:3.10-slim as the base
FROM python:3.10-slim

# Accept an ARG RUN_ID as specified
ARG RUN_ID
ENV MLFLOW_RUN_ID=$RUN_ID

# Include a command to "download" the model (Simulated)
RUN echo "Downloading model from MLflow using Run ID: ${MLFLOW_RUN_ID}"

# Default command
CMD ["echo", "Model Container is prepared!"]
