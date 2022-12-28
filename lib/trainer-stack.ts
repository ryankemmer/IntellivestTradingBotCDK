import { Stack, StackProps, Duration } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import {
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_lambda as lambda
  } from "aws-cdk-lib";

export class TrainerStack extends Stack {
    constructor(scope: Construct, id: string, props?: StackProps) {
        super(scope, id, props);
        
        const vpc = new ec2.Vpc(this, "VPC", {
            natGateways: 1,
            maxAzs: 1
          });

        // ECS cluster to deploy tasks to
        const cluster = new ecs.Cluster(this, "ECSCluster", {
            clusterName: `${props?.stackName}-cluster`,
            vpc,
            containerInsights: true // allow metrics to show up in cloudwath
        });

        // ecs task for defining config 
        const ECSTaskRunner = new ecs.TaskDefinition(this, `ECSTaskDef`, {
            family: `${props?.stackName}-task-definition`,
            compatibility: ecs.Compatibility.FARGATE,
            // provide the memory and cpu needed for the task
            cpu: "1024",
            memoryMiB: "2048"
        });
  
        // add any permissions/policies that your task may need e.g "s3:PutObject"
        ECSTaskRunner.addToTaskRolePolicy(new iam.PolicyStatement({
            actions: ["ecs:StartTelemetrySession"] as string[],
            effect: iam.Effect.ALLOW,
            resources: ["*"]
        }));

        const container = ECSTaskRunner.addContainer(`ECSContainer`, {
            containerName: `${props?.stackName}-container`,
            image: ecs.ContainerImage.fromAsset('src/trainer'),
            environment: {
                NORMAL_ENV_VAR: "example"
            },
            logging: ecs.LogDriver.awsLogs({ streamPrefix: `${props?.stackName}-container-logs` })
        });

        // give the lambda functions access to trigger the tasks
        const lambdaRoleTrigger = new iam.Role(this, "lambda-role", {
            assumedBy: new iam.AnyPrincipal(),
            inlinePolicies: {
            "inline-lambda-trigger-policy": new iam.PolicyDocument({
                statements: [new iam.PolicyStatement({
                effect: iam.Effect.ALLOW,
                resources: [ECSTaskRunner.taskDefinitionArn],
                actions: ["ecs:RunTask"]
                }),
                new iam.PolicyStatement({
                effect: iam.Effect.ALLOW,
                resources: ["*"],
                actions: ["iam:PassRole"]
                }),
                new iam.PolicyStatement({
                effect: iam.Effect.ALLOW,
                resources: [cluster.clusterArn],
                actions: ["ecs:DescribeTasks"]
                })
                ]
            })
            }
        });

        // get the subnet id values from the vpc
        const subnets = vpc.selectSubnets({
            subnetType: ec2.SubnetType.PRIVATE_WITH_NAT
        }).subnets;
    
        //Create lambda function for closing trades
        const triggerLambda = new lambda.DockerImageFunction(this, 'trainTriggerLambda', {
            functionName: "trainTriggerLambda",
            memorySize: 500,
            timeout: Duration.minutes(1),
            code: lambda.DockerImageCode.fromImageAsset('src/triggerfunction'),
            environment: {
                ECS_TASK_ARN: ECSTaskRunner.taskDefinitionArn,
                ECS_CLUSTER_ARN: cluster.clusterArn,
                SUBNET_IDS: subnets.map(sub => sub.subnetId).join(","),
                ECS_CONTAINER_NAME: container.containerName
              },
            role: lambdaRoleTrigger,
        });


    };
}