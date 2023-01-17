# Intellivest Trading Bot Infrastrcture

Full stack infastructure to host automatic trading bot strategies. Built using AWS CDK.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

## Updating this project

* Edit python lambda code in `lambda` folder to update buying/ selling logic
* Edit python code in `trainer` folder to update trainer logic
* Run `pipreqs src/lambda --force` and `pipreqs src/trainer --force` in the root directory of the project to update Python dependencies
* <b>Optional</b> - test to make sure container will build

```
docker build -t src/buyingfunction .
docker run -p 9000:8080 functions

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'

```


* `npm run build`   compile typescript to js
* `cdk synth`       synthesize couldformation stack
* `cdk deploy`      deploy this stack to your default AWS account/region


## Other Useful commands

* `cdk diff`        compare deployed stack with current state
* `cdk synth`       emits the synthesized CloudFormation template
