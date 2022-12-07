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

    const lambdaARole = new Role(this, 'LambdaRole', {
      assumedBy: new ServicePrincipal('lambda.amazonaws.com'),
    });

    lambdaARole.addManagedPolicy(
      ManagedPolicy.fromAwsManagedPolicyName('AmazonDynamoDBFullAccess')
    );
    
    //Create lambda function
    const newLambda = new lambda.DockerImageFunction(this, 'scheduledLambda',{
      functionName: "tradinglambda",
      memorySize: 500,
      timeout: Duration.seconds(30),
      code: lambda.DockerImageCode.fromImageAsset('functions'),
      role: lambdaARole,
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
    const tradestable = new dynamodb.Table(this, 'TradesTable', {
      tableName: 'tradestable',
      partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
    });

    //Create table of historical trades 
    const historicaltable = new dynamodb.Table(this, 'HistoricTradesTable', {
      tableName: 'historicaltradestable',
      partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
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

    newLambda.addToRolePolicy(
      new PolicyStatement({
        actions: [
          's3:ListBucket'],
        resources: [
          bucket.bucketArn],
      })
    );

    newLambda.addToRolePolicy(
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
