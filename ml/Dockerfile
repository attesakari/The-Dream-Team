FROM python:3.13-slim

RUN pip install pipenv

WORKDIR /app

# COPY ["requirements.txt", "./"]
COPY ["Pipfile", "Pipfile.lock", "./"]
# RUN pipenv install -r requirements.txt
RUN pipenv install --system --deploy

# COPY ["*.py", "churn_model.pkl", "./"]
# COPY ["*.py", "./"]
COPY ["./", "./"]

EXPOSE 9696

ENTRYPOINT ["python", "-m", "gunicorn", "--bind=0.0.0.0:9696", "churn_predict:app"]
