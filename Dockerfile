# Dockerfile
# Use a specific, stable version of Python
FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Install system dependencies needed for some Python packages or health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install and configure Poetry
RUN pip install --no-cache-dir poetry
# This tells poetry to install dependencies in the system's python, not a venv
RUN poetry config virtualenvs.create false

# Copy the dependency definition files from the 'app' subdirectory
COPY app/pyproject.toml app/poetry.lock* ./

# --- THIS IS THE FINAL FIX ---
# Add the --no-root flag to only install dependencies, not the project itself.
RUN poetry install --no-root --without dev --no-interaction --no-ansi

# Copy the rest of the application source code
COPY ./app /code/app
COPY ./scripts /code/scripts
COPY ./data /code/data

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
