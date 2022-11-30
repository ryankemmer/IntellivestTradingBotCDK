import { Stack, StackProps} from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { aws_lambda as lambda } from 'aws-cdk-lib';
import { aws_events_targets as targets } from 'aws-cdk-lib';
import { aws_events as events } from 'aws-cdk-lib';
import { aws_dynamodb as dynamodb } from 'aws-cdk-lib';
import { aws_s3 as s3 } from 'aws-cdk-lib';

export class IntellivestTradingBotCdkStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);
    
    //Create lambda function
    const newLambda = new lambda.DockerImageFunction(this, 'scheduledLambda',{
      code: lambda.DockerImageCode.fromImageAsset('functions')
    });

    // Run every weekday day at 1:30PM UTC
    // See https://docs.aws.amazon.com/lambda/latest/dg/tutorial-scheduled-events-schedule-expressions.html
    const eventRule = new events.Rule(this, 'scheduleRule', {
      schedule: events.Schedule.cron({ 
        minute: '30', 
        hour: '13',
        month: '*', 
        weekDay: 'MON-FRI', 
        year: '*'
        }),
    });
    eventRule.addTarget(new targets.LambdaFunction(newLambda))

    //Create DynamoDB to store trades
    const table = new dynamodb.Table(this, 'TradesTable', {
      partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
    });

    //Create s3 Bucket to store models and artifacts
    const bucket = new s3.Bucket(this, 'ArtifactsBucket');
  }
}
