# ISD: Real-Time AI-powered Safety Gear Detection System

ISD is an AI-powered computer vision system designed to monitor, track, and enforce workplace safety gear compliance. It utilizes real-time image analysis to identify five essential safety items: helmet, gloves, jacket, goggles, and footwear. ISD verifies if personnel are wearing these items before granting access to the work area. For implementation of Industry saftey Yolo7 model of computer vision is used, which has smaller size. 

- Yolov7 Github repo link (https://github.com/WongKinYiu/yolov7)

# Flow Diaglram of MLops pipeline

![MLops_ISD_Architexture_Flow_Diagram drawio](https://github.com/data-pioneer/MLops-Industry-Safety-Detection-using-Yolov7/assets/33811437/23cb87e0-2184-4a27-8189-13343207539b)


# Benefits

- Enhances workplace safety by ensuring proper safety gear usage.
- Improves efficiency with real-time monitoring and automated access control.

## Workflows

 - constants
 - config_entity
 - artifact_entity
 - components
 - pipeline
 - app.py


## Git commands

```bash
git add .

git commit -m "Updated"

git push origin main
```


## AWS Configurations

```bash
#aws cli download link: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

aws configure
```


## How to run?

```bash
conda create -n safety python=3.8 -y
```

```bash
conda activate safety
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```


# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 136566696263.dkr.ecr.us-east-1.amazonaws.com/yolov7app

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app
