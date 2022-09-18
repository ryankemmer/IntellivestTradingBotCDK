import * as cdk from '@aws-cdk/core';
import * as lambda from '@aws-cdk/aws-lambda';
import * as targets from '@aws-cdk/aws-events-targets'
import * as events from '@aws-cdk/aws-events'

export class IntellivestTradingBotCdkStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

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
  }
}
