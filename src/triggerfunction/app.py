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


def stop_task(event, context):
    logger.info(event)

    ecs = boto3.client('ecs')

    cluster = os.environ.get('ECS_CLUSTER_ARN'),
    tasks = ecs.list_tasks(cluster=cluster)

    if (tasks is None):
        return "No tasks to stop"

    tasksResponse = ecs.describe_tasks(
        cluster=cluster,
        tasks= tasks.taskArns
    )

    totalStopped = 0
    if tasksResponse is not None:
        for task in tasksResponse.tasks:
            
            parts = tasks.taskArn.split("/")
            taskId = parts[len(parts) - 1]
            

    


