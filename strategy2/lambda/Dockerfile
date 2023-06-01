FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.8

#copy source 
COPY source/auth_params.py source/features.py source/ModelLoaderS3.py source/TimeSeriesPrep.py source/yfinanceDAO.py ${LAMBDA_TASK_ROOT}

# Copy function code
COPY buyingFunction.py sellingFunction.py triggerFunction.py ${LAMBDA_TASK_ROOT}

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
