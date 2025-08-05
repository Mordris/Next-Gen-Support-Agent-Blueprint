# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory to a neutral location
WORKDIR /code

# 3. Install Poetry
RUN pip install poetry

# 4. Copy only the dependency definitions to leverage Docker layer caching
# We copy them to the WORKDIR, which is now /code
COPY app/pyproject.toml app/poetry.lock* ./

# 5. Install project dependencies
RUN poetry install --without dev --no-root

# 6. Copy the rest of the application source code into a subdirectory
COPY ./app /code/app

# 7. Command to run the application (This stays the same)
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
