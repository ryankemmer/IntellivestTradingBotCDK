#!/usr/bin/env node
import { App } from 'aws-cdk-lib';
import { IntellivestTradingBotCdkStack } from '../lib/intellivest_trading_bot_cdk-stack';
import { TrainerStack } from '../lib/trainer-stack';

const app = new App();
new IntellivestTradingBotCdkStack(app, 'IntellivestTradingBotCdkStack', {});
new TrainerStack(app, 'TrainerStack', {stackName: 'TrainerStack'})