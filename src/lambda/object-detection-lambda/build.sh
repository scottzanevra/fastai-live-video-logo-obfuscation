#!/bin/bash
#
set -o nounset -o errexit -o posix

# needed to avoid issues with virtualenv + nounset
export VIRTUAL_ENV_DISABLE_PROMPT="true"

# the lambda function name
export FunctionName="fast-ai-object-detection-lambda"
export DeploymentRegion='ap-southeast-2'
export CapabilityName='CAPABILITY_IAM'
export S3ArtifactBucket="scott-testing-timeseries"

APP_ROOT='./'

pushd "${APP_ROOT}"

echo "installed dependency"
PIP_TRUSTED_HOST="pypi.python.org pypi.org files.pythonhosted.org devpi"

sam build --manifest "${APP_ROOT}/app/requirements.txt" --build-dir ./build

sam deploy \
    --no-fail-on-empty-changeset \
    --capabilities ${CapabilityName} \
    --template-file ./template.yaml\
    --stack-name "${FunctionName}-stack" \
    --s3-bucket "${S3ArtifactBucket}" \
    --s3-prefix "scott/${FunctionName}" \
    --region ${DeploymentRegion} \
    --tags "Course=FastAI" "Project=ObjectDetection" \

popd

