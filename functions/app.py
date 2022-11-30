import logging
from functools import reduce

def handler(event, context):

    print('lets start')
    return {
        'statusCode': 200,
        'body': json.dumps(f'Done! Recorded')
    }