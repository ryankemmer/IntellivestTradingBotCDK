FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.8

# Copy function code
COPY app.py auth_params.py ModelLoaderS3.py features.py yfinanceDAO.py TimeSeriesPrep.py ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]