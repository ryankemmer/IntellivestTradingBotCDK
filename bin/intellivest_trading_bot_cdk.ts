#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from '@aws-cdk/core';
import { App } from 'aws-cdk-lib';
import { IntellivestTradingBotCdkStack } from '../lib/intellivest_trading_bot_cdk-stack';

const app = new cdk.App();
new IntellivestTradingBotCdkStack(app, 'IntellivestTradingBotCdkStack', {

  /* AWS Account and Region that are implied by the current CLI configuration. */
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },

  /* For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html */
});