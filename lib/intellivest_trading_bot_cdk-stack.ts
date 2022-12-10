import { Stack, StackProps, Duration} from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { aws_lambda as lambda } from 'aws-cdk-lib';
import { aws_events_targets as targets } from 'aws-cdk-lib';
import { aws_events as events } from 'aws-cdk-lib';
import { aws_dynamodb as dynamodb } from 'aws-cdk-lib';
import { aws_s3 as s3 } from 'aws-cdk-lib';
import { Effect, ManagedPolicy, PolicyStatement, ServicePrincipal, Role } from 'aws-cdk-lib/aws-iam';

export class IntellivestTradingBotCdkStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    //Create Role for Dynamo
    const lambdaRole = new Role(this, 'LambdaRole', {
      assumedBy: new ServicePrincipal('lambda.amazonaws.com'),
    });

    lambdaRole.addManagedPolicy(
      ManagedPolicy.fromAwsManagedPolicyName('AmazonDynamoDBFullAccess')
    );
    
    //Create lambda function for opening new trades
    const buyingLambda = new lambda.DockerImageFunction(this, 'buyingLambda',{
      functionName: "buyinglambda",
      memorySize: 500,
      timeout: Duration.minutes(10),
      code: lambda.DockerImageCode.fromImageAsset('buyingfunction'),
      role: lambdaRole,
    });

    // Run every weekday day at 1:30PM UTC
    // See https://docs.aws.amazon.com/lambda/latest/dg/tutorial-scheduled-events-schedule-expressions.html
    const eventRuleDaily = new events.Rule(this, 'buyingSchedule', {
      schedule: events.Schedule.cron({ 
        minute: '30', 
        hour: '20',
        month: '*', 
        weekDay: 'MON-FRI', 
        year: '*'
        }),
    });
    eventRuleDaily.addTarget(new targets.LambdaFunction(buyingLambda))

    //Create lambda function for closing trades
    const sellingLambda = new lambda.DockerImageFunction(this, 'sellingLambda',{
      functionName: "sellinglambda",
      memorySize: 500,
      timeout: Duration.minutes(1),
      code: lambda.DockerImageCode.fromImageAsset('sellingfunction'),
      role: lambdaRole,
    });

    // Run every 30 seconds in open trading hours (starting 30 minutes early)
    const eventRuleMonitor = new events.Rule(this, 'monitorSchedule', {
      schedule: events.Schedule.cron({ 
        minute: '*', 
        hour: '9-17',
        month: '*', 
        weekDay: 'MON-FRI', 
        year: '*'
        }),
    });
    eventRuleMonitor.addTarget(new targets.LambdaFunction(sellingLambda))

    //Create DynamoDB to store trades
    const tradestable = new dynamodb.Table(this, 'TradesTable', {
      tableName: 'tradestable',
      partitionKey: { name: 'date', type: dynamodb.AttributeType.STRING },
    });

    //Create table of historical trades 
    const historicaltable = new dynamodb.Table(this, 'HistoricTradesTable', {
      tableName: 'historicaltradestable',
      partitionKey: { name: 'date', type: dynamodb.AttributeType.STRING },
    });

    //Create s3 Bucket to store models and artifacts
    const bucket = new s3.Bucket(this, 'ArtifactsBucket', {
      versioned: false,
      bucketName: "artifactsbucketintellivest",
      publicReadAccess: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    });

    bucket.addToResourcePolicy(
      new PolicyStatement({
        effect: Effect.ALLOW,
        principals: [new ServicePrincipal('lambda.amazonaws.com')],
        actions: [
          's3:ListBucket'],
        resources: [
          bucket.bucketArn],
      }),
    );

    bucket.addToResourcePolicy(
      new PolicyStatement({
        effect: Effect.ALLOW,
        principals: [new ServicePrincipal('lambda.amazonaws.com')],
        actions: [ 
          's3:GetObject', 
          's3:PutObject', 
          's3:DeleteObject',],
        resources: [`${bucket.bucketArn}/*`],
      })
    );

    buyingLambda.addToRolePolicy(
      new PolicyStatement({
        actions: [
          's3:ListBucket'],
        resources: [
          bucket.bucketArn],
      })
    );

    buyingLambda.addToRolePolicy(
      new PolicyStatement({
        actions: [
          's3:GetObject', 
          's3:PutObject', 
          's3:DeleteObject'],
        resources: [
          `${bucket.bucketArn}/*`],
      })
    );

  }
}
