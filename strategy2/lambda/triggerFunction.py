import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import boto3
import os

def handler(event, context):
    logger.info(event)

    ecs = boto3.client('ecs')

    response = ecs.run_task(
        cluster= os.environ.get('ECS_CLUSTER_ARN'),
		taskDefinition= os.environ.get('ECS_TASK_ARN'),
		networkConfiguration= {
			'awsvpcConfiguration': {
				'subnets': os.environ.get('SUBNET_IDS').split(","),
				'assignPublicIp': "DISABLED"
			}
		},
		count= 1,
		launchType= "FARGATE"
    )

    logger.info(response)

    return "Started task"



