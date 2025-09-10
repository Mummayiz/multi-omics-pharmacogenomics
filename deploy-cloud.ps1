# Multi-Omics Pharmacogenomics Platform - Cloud Deployment Script
# PowerShell script for AWS/GCP deployment

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("aws", "gcp", "azure")]
    [string]$CloudProvider,
    
    [string]$Environment = "production",
    [string]$Region = "",
    [string]$InstanceType = "m5.xlarge",
    [string]$KeyName = "",
    [switch]$CreateInfrastructure = $false,
    [switch]$DeployOnly = $false
)

Write-Host "🌩️ Multi-Omics Pharmacogenomics Platform - Cloud Deployment" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green

$ErrorActionPreference = "Stop"

# Set default regions
if (-not $Region) {
    switch ($CloudProvider) {
        "aws" { $Region = "us-east-1" }
        "gcp" { $Region = "us-central1" }
        "azure" { $Region = "East US" }
    }
}

Write-Host "🔧 Configuration:" -ForegroundColor Blue
Write-Host "   Cloud Provider: $CloudProvider" -ForegroundColor Gray
Write-Host "   Environment: $Environment" -ForegroundColor Gray
Write-Host "   Region: $Region" -ForegroundColor Gray
Write-Host "   Instance Type: $InstanceType" -ForegroundColor Gray

# Function to deploy to AWS
function Deploy-AWS {
    Write-Host "🚀 Deploying to AWS..." -ForegroundColor Blue
    
    # Check AWS CLI
    try {
        aws --version | Out-Null
        Write-Host "✅ AWS CLI is available" -ForegroundColor Green
    } catch {
        Write-Host "❌ AWS CLI is not installed. Please install AWS CLI." -ForegroundColor Red
        return
    }

    if ($CreateInfrastructure) {
        Write-Host "🏗️ Creating AWS infrastructure..." -ForegroundColor Blue
        
        # Create CloudFormation template for basic infrastructure
        $cfTemplate = @"
{
    "AWSTemplateFormatVersion": "2010-09-09",
    "Description": "Multi-Omics Pharmacogenomics Platform Infrastructure",
    "Parameters": {
        "InstanceType": {
            "Type": "String",
            "Default": "$InstanceType",
            "Description": "EC2 instance type"
        },
        "KeyName": {
            "Type": "String",
            "Default": "$KeyName",
            "Description": "EC2 Key Pair name"
        }
    },
    "Resources": {
        "MultiOmicsVPC": {
            "Type": "AWS::EC2::VPC",
            "Properties": {
                "CidrBlock": "10.0.0.0/16",
                "EnableDnsHostnames": true,
                "EnableDnsSupport": true,
                "Tags": [
                    {"Key": "Name", "Value": "MultiOmics-VPC"}
                ]
            }
        },
        "PublicSubnet": {
            "Type": "AWS::EC2::Subnet",
            "Properties": {
                "VpcId": {"Ref": "MultiOmicsVPC"},
                "CidrBlock": "10.0.1.0/24",
                "AvailabilityZone": {"Fn::Select": [0, {"Fn::GetAZs": ""}]},
                "MapPublicIpOnLaunch": true,
                "Tags": [
                    {"Key": "Name", "Value": "MultiOmics-PublicSubnet"}
                ]
            }
        },
        "InternetGateway": {
            "Type": "AWS::EC2::InternetGateway",
            "Properties": {
                "Tags": [
                    {"Key": "Name", "Value": "MultiOmics-IGW"}
                ]
            }
        },
        "AttachGateway": {
            "Type": "AWS::EC2::VPCGatewayAttachment",
            "Properties": {
                "VpcId": {"Ref": "MultiOmicsVPC"},
                "InternetGatewayId": {"Ref": "InternetGateway"}
            }
        },
        "SecurityGroup": {
            "Type": "AWS::EC2::SecurityGroup",
            "Properties": {
                "GroupDescription": "Security group for Multi-Omics Platform",
                "VpcId": {"Ref": "MultiOmicsVPC"},
                "SecurityGroupIngress": [
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 80,
                        "ToPort": 80,
                        "CidrIp": "0.0.0.0/0"
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 443,
                        "ToPort": 443,
                        "CidrIp": "0.0.0.0/0"
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 8000,
                        "ToPort": 8000,
                        "CidrIp": "0.0.0.0/0"
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "CidrIp": "0.0.0.0/0"
                    }
                ]
            }
        },
        "EC2Instance": {
            "Type": "AWS::EC2::Instance",
            "Properties": {
                "ImageId": "ami-0c02fb55956c7d316",
                "InstanceType": {"Ref": "InstanceType"},
                "KeyName": {"Ref": "KeyName"},
                "SecurityGroupIds": [{"Ref": "SecurityGroup"}],
                "SubnetId": {"Ref": "PublicSubnet"},
                "UserData": {
                    "Fn::Base64": {
                        "Fn::Join": ["", [
                            "#!/bin/bash\n",
                            "yum update -y\n",
                            "yum install -y docker\n",
                            "service docker start\n",
                            "usermod -a -G docker ec2-user\n",
                            "curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose\n",
                            "chmod +x /usr/local/bin/docker-compose\n"
                        ]]
                    }
                },
                "Tags": [
                    {"Key": "Name", "Value": "MultiOmics-Server"}
                ]
            }
        }
    },
    "Outputs": {
        "InstanceId": {
            "Description": "EC2 Instance ID",
            "Value": {"Ref": "EC2Instance"}
        },
        "PublicIP": {
            "Description": "Public IP address",
            "Value": {"Fn::GetAtt": ["EC2Instance", "PublicIp"]}
        }
    }
}
"@
        
        $cfTemplate | Out-File -FilePath "aws-infrastructure.json" -Encoding UTF8
        
        # Deploy CloudFormation stack
        $stackName = "multi-omics-platform-$Environment"
        Write-Host "   Creating CloudFormation stack: $stackName" -ForegroundColor Gray
        
        aws cloudformation deploy `
            --template-file aws-infrastructure.json `
            --stack-name $stackName `
            --parameter-overrides InstanceType=$InstanceType KeyName=$KeyName `
            --region $Region `
            --capabilities CAPABILITY_IAM

        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Infrastructure created successfully" -ForegroundColor Green
            
            # Get instance details
            $outputs = aws cloudformation describe-stacks --stack-name $stackName --region $Region --query "Stacks[0].Outputs" --output json | ConvertFrom-Json
            $instanceId = ($outputs | Where-Object { $_.OutputKey -eq "InstanceId" }).OutputValue
            $publicIp = ($outputs | Where-Object { $_.OutputKey -eq "PublicIP" }).OutputValue
            
            Write-Host "   Instance ID: $instanceId" -ForegroundColor Gray
            Write-Host "   Public IP: $publicIp" -ForegroundColor Gray
            
            # Wait for instance to be ready
            Write-Host "⏳ Waiting for instance to be ready..." -ForegroundColor Blue
            aws ec2 wait instance-running --instance-ids $instanceId --region $Region
            Start-Sleep -Seconds 60  # Additional time for user data script
            
            Write-Host "✅ Instance is ready" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to create infrastructure" -ForegroundColor Red
            return
        }
    }
    
    if (-not $DeployOnly -or $CreateInfrastructure) {
        Write-Host "📦 Deploying application..." -ForegroundColor Blue
        Write-Host "   Use SSH to connect to your instance and deploy the application" -ForegroundColor Gray
        Write-Host "   ssh -i your-key.pem ec2-user@$publicIp" -ForegroundColor Gray
    }
}

# Function to deploy to GCP
function Deploy-GCP {
    Write-Host "🚀 Deploying to Google Cloud Platform..." -ForegroundColor Blue
    
    # Check gcloud CLI
    try {
        gcloud version | Out-Null
        Write-Host "✅ gcloud CLI is available" -ForegroundColor Green
    } catch {
        Write-Host "❌ gcloud CLI is not installed. Please install Google Cloud SDK." -ForegroundColor Red
        return
    }

    if ($CreateInfrastructure) {
        Write-Host "🏗️ Creating GCP infrastructure..." -ForegroundColor Blue
        
        # Create deployment script for GCP
        $gcpDeployScript = @"
#!/bin/bash
# GCP Deployment Script for Multi-Omics Platform

PROJECT_ID=`$(gcloud config get-value project)
ZONE="$Region-a"
INSTANCE_NAME="multi-omics-platform-$Environment"

# Create Compute Engine instance
gcloud compute instances create `$INSTANCE_NAME \
    --zone=`$ZONE \
    --machine-type=n1-standard-4 \
    --subnet=default \
    --network-tier=PREMIUM \
    --maintenance-policy=MIGRATE \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-standard \
    --tags=http-server,https-server,multi-omics \
    --metadata-from-file startup-script=startup-script.sh

# Create firewall rules
gcloud compute firewall-rules create multi-omics-http \
    --allow tcp:80,tcp:443,tcp:8000 \
    --source-ranges 0.0.0.0/0 \
    --target-tags multi-omics

echo "Instance created: `$INSTANCE_NAME"
gcloud compute instances describe `$INSTANCE_NAME --zone=`$ZONE --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
"@
        
        $gcpDeployScript | Out-File -FilePath "deploy-gcp.sh" -Encoding UTF8
        
        # Create startup script
        $startupScript = @"
#!/bin/bash
apt-get update
apt-get install -y docker.io docker-compose git
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Clone repository (you'll need to update this with your actual repo)
cd /home/ubuntu
# git clone https://github.com/mohdmum/multi-omics-pharmacogenomics-platform.git
# cd multi-omics-pharmacogenomics-platform
# docker-compose up -d
"@
        
        $startupScript | Out-File -FilePath "startup-script.sh" -Encoding UTF8
        
        # Make scripts executable and run
        Write-Host "   Running GCP deployment script..." -ForegroundColor Gray
        bash deploy-gcp.sh
    }
}

# Main deployment logic
switch ($CloudProvider) {
    "aws" { Deploy-AWS }
    "gcp" { Deploy-GCP }
    "azure" { 
        Write-Host "🔵 Azure deployment coming soon..." -ForegroundColor Blue
        Write-Host "   Please use the Docker deployment method for now." -ForegroundColor Gray
    }
    default {
        Write-Host "❌ Unsupported cloud provider: $CloudProvider" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Cloud deployment script completed!" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
