# ══════════════════════════════════════════════════════════════════════════════
# AWS Deployment Guide — HR Policy RAG Chatbot
# ══════════════════════════════════════════════════════════════════════════════
# Replace all YOUR_* placeholders with your actual values before running.
# Run commands in order — each step depends on the previous one.
# ══════════════════════════════════════════════════════════════════════════════

# ── Prerequisites ─────────────────────────────────────────────────────────────
# 1. AWS CLI installed and configured (aws configure)
# 2. Docker installed and running
# 3. Your AWS account ID ready (12-digit number)

export AWS_ACCOUNT_ID=YOUR_ACCOUNT_ID        # e.g. 123456789012
export AWS_REGION=YOUR_REGION                # e.g. ap-south-1
export ECR_REPO=hr-rag-chatbot
export ECS_CLUSTER=hr-rag-cluster
export ECS_SERVICE_API=hr-rag-api-service
export ECS_SERVICE_UI=hr-rag-streamlit-service


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Create ECR Repository
# ECR = Elastic Container Registry (stores your Docker images)
# ══════════════════════════════════════════════════════════════════════════════

aws ecr create-repository \
  --repository-name $ECR_REPO \
  --region $AWS_REGION \
  --image-scanning-configuration scanOnPush=true

# Note the repositoryUri from the output:
# YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/hr-rag-chatbot


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build and Push Docker Image to ECR
# ══════════════════════════════════════════════════════════════════════════════

ECR_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_URI

# Build image
docker build -t $ECR_REPO:latest .

# Tag and push
docker tag $ECR_REPO:latest $ECR_URI:latest
docker push $ECR_URI:latest


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Store Secrets in AWS Secrets Manager
# Never put API keys in ECS task definitions directly
# ══════════════════════════════════════════════════════════════════════════════

aws secretsmanager create-secret \
  --name hr-rag/groq-api-key \
  --secret-string "your_actual_groq_api_key_here" \
  --region $AWS_REGION

# Note the ARN — paste it into ecs-task-api.json and ecs-task-streamlit.json:
# arn:aws:secretsmanager:YOUR_REGION:YOUR_ACCOUNT_ID:secret:hr-rag/groq-api-key


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Create EFS for Persistent Storage
# EFS = Elastic File System (shared volume for vectorstore + HR docs)
# Both API and Streamlit containers mount this
# ══════════════════════════════════════════════════════════════════════════════

# Create EFS file system
aws efs create-file-system \
  --creation-token hr-rag-efs \
  --encrypted \
  --region $AWS_REGION \
  --tags Key=Name,Value=hr-rag-efs

# Note the FileSystemId (fs-xxxxxxxx) — replace YOUR_EFS_ID in task definitions

# Create access point for vectorstore
aws efs create-access-point \
  --file-system-id YOUR_EFS_ID \
  --posix-user Uid=1000,Gid=1000 \
  --root-directory "Path=/vectorstore,CreationInfo={OwnerUid=1000,OwnerGid=1000,Permissions=755}" \
  --region $AWS_REGION

# Note the AccessPointId — replace YOUR_EFS_ACCESS_POINT_ID in task definitions


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Create IAM Roles
# ══════════════════════════════════════════════════════════════════════════════

# ECS Task Execution Role (lets ECS pull images + read secrets)
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Allow reading secrets
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite

# ECS Task Role (permissions the app itself needs)
aws iam create-role \
  --role-name ecsTaskRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Allow EFS access from task
aws iam attach-role-policy \
  --role-name ecsTaskRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonElasticFileSystemClientReadWriteAccess


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Create ECS Cluster
# ══════════════════════════════════════════════════════════════════════════════

aws ecs create-cluster \
  --cluster-name $ECS_CLUSTER \
  --capacity-providers FARGATE \
  --region $AWS_REGION


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Register ECS Task Definitions
# First update ecs-task-api.json and ecs-task-streamlit.json with your values:
#   YOUR_ACCOUNT_ID → your 12-digit account ID
#   YOUR_REGION     → e.g. ap-south-1
#   YOUR_EFS_ID     → from Step 4
#   YOUR_EFS_ACCESS_POINT_ID → from Step 4
# ══════════════════════════════════════════════════════════════════════════════

aws ecs register-task-definition \
  --cli-input-json file://infra/ecs-task-api.json \
  --region $AWS_REGION

aws ecs register-task-definition \
  --cli-input-json file://infra/ecs-task-streamlit.json \
  --region $AWS_REGION


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Create ECS Services
# Fargate launches tasks as serverless containers (no EC2 to manage)
# Replace YOUR_SUBNET_ID and YOUR_SECURITY_GROUP_ID with your VPC values
# ══════════════════════════════════════════════════════════════════════════════

# API Service
aws ecs create-service \
  --cluster $ECS_CLUSTER \
  --service-name $ECS_SERVICE_API \
  --task-definition hr-rag-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[YOUR_SUBNET_ID],
    securityGroups=[YOUR_SECURITY_GROUP_ID],
    assignPublicIp=ENABLED
  }" \
  --region $AWS_REGION

# Streamlit Service
aws ecs create-service \
  --cluster $ECS_CLUSTER \
  --service-name $ECS_SERVICE_UI \
  --task-definition hr-rag-streamlit \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[YOUR_SUBNET_ID],
    securityGroups=[YOUR_SECURITY_GROUP_ID],
    assignPublicIp=ENABLED
  }" \
  --region $AWS_REGION


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Ingest HR Documents (run once after deployment)
# ══════════════════════════════════════════════════════════════════════════════

# Call the ingest endpoint on your deployed API
curl -X POST https://YOUR_API_URL/ingest \
  -H "Content-Type: application/json" \
  -d '{"reset": false}'


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Set Up CI/CD with CodeBuild (optional but recommended)
# ══════════════════════════════════════════════════════════════════════════════

# 1. Go to AWS CodeBuild → Create build project
# 2. Source: Connect your GitHub repo
# 3. Buildspec: Use buildspec.yml file in repo
# 4. Environment variables to add in CodeBuild console:
#    AWS_ACCOUNT_ID   = your account ID
#    AWS_REGION       = your region
#    ECR_REPO_NAME    = hr-rag-chatbot
#    ECS_CLUSTER      = hr-rag-cluster
#    ECS_SERVICE_API  = hr-rag-api-service
#    ECS_SERVICE_UI   = hr-rag-streamlit-service
# 5. Give CodeBuild role permissions to push ECR + update ECS
# 6. Every git push to main triggers a build → deploy


# ══════════════════════════════════════════════════════════════════════════════
# Useful Commands After Deployment
# ══════════════════════════════════════════════════════════════════════════════

# Check service status
aws ecs describe-services \
  --cluster $ECS_CLUSTER \
  --services $ECS_SERVICE_API \
  --region $AWS_REGION

# View logs in CloudWatch
aws logs tail /ecs/hr-rag-api --follow --region $AWS_REGION

# Force re-deploy (after pushing new image to ECR)
aws ecs update-service \
  --cluster $ECS_CLUSTER \
  --service $ECS_SERVICE_API \
  --force-new-deployment \
  --region $AWS_REGION
