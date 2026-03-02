"""
Fate RAG CDK Stack
Provisions:
  - S3 bucket for raw documents
  - OpenSearch Serverless collection (vector search)
  - Lambda function (FastAPI backend via Mangum)
  - API Gateway HTTP API
  - IAM roles with least-privilege policies
"""

from __future__ import annotations

import json

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_opensearchserverless as aoss
from aws_cdk import aws_s3 as s3
from constructs import Construct


class FateRagStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env_name: str = "dev",
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        is_prod = env_name == "prod"

        # ── S3: Raw document storage ─────────────────────────────────────────
        self.documents_bucket = s3.Bucket(
            self,
            "DocumentsBucket",
            bucket_name=f"fate-rag-documents-{env_name}-{self.account}",
            versioned=is_prod,
            removal_policy=RemovalPolicy.RETAIN if is_prod else RemovalPolicy.DESTROY,
            auto_delete_objects=not is_prod,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-raw-after-90-days",
                    prefix="raw/",
                    expiration=Duration.days(90),
                )
            ],
        )

        # ── OpenSearch Serverless ────────────────────────────────────────────
        collection_name = f"fate-lore-{env_name}"

        # Encryption policy
        encryption_policy = aoss.CfnSecurityPolicy(
            self,
            "EncryptionPolicy",
            name=f"fate-rag-enc-{env_name}",
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{collection_name}"],
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
        )

        # Network policy (public for dev, VPC for prod ideally)
        network_policy = aoss.CfnSecurityPolicy(
            self,
            "NetworkPolicy",
            name=f"fate-rag-net-{env_name}",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                            },
                            {
                                "ResourceType": "dashboard",
                                "Resource": [f"collection/{collection_name}"],
                            },
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
        )

        # OpenSearch Serverless collection
        self.collection = aoss.CfnCollection(
            self,
            "FateLoreCollection",
            name=collection_name,
            type="VECTORSEARCH",
            description="Fate Series lore vector store for RAG",
        )
        self.collection.add_dependency(encryption_policy)
        self.collection.add_dependency(network_policy)

        # ── IAM: Lambda execution role ────────────────────────────────────────
        lambda_role = iam.Role(
            self,
            "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Bedrock permissions (Titan embeddings + Claude generation)
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockInvokeModels",
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v1",
                    f"arn:aws:bedrock:{self.region}::foundation-model/claude-sonnet-4-20250514",
                    f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-*",
                ],
            )
        )

        # OpenSearch Serverless permissions
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="OpenSearchServerlessAccess",
                actions=["aoss:APIAccessAll"],
                resources=[self.collection.attr_arn],
            )
        )

        # S3 read permissions for document bucket
        self.documents_bucket.grant_read(lambda_role)

        # ── OpenSearch data access policy ─────────────────────────────────────
        aoss.CfnAccessPolicy(
            self,
            "DataAccessPolicy",
            name=f"fate-rag-access-{env_name}",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "index",
                                "Resource": [f"index/{collection_name}/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                    "aoss:UpdateIndex",
                                    "aoss:DescribeIndex",
                                ],
                            },
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                                "Permission": ["aoss:DescribeCollectionItems"],
                            },
                        ],
                        "Principal": [lambda_role.role_arn],
                    }
                ]
            ),
        )

        # ── Lambda function ───────────────────────────────────────────────────
        self.backend_function = lambda_.Function(
            self,
            "BackendFunction",
            function_name=f"fate-rag-backend-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="backend.app.handler",
            code=lambda_.Code.from_asset(
                ".",
                exclude=[
                    ".git",
                    ".github",
                    "__pycache__",
                    "*.pyc",
                    "infra",
                    "data",
                    "notebooks",
                    "tests",
                    "frontend",
                    ".env*",
                    "docker-compose.yml",
                    "cdk.out",
                ],
            ),
            role=lambda_role,
            timeout=Duration.seconds(60),
            memory_size=512,
            environment={
                "AWS_REGION_NAME": self.region,
                "BEDROCK_REGION": self.region,
                "BEDROCK_MODEL_ID": "claude-sonnet-4-20250514",
                "BEDROCK_EMBEDDING_MODEL": "amazon.titan-embed-text-v1",
                "OPENSEARCH_ENDPOINT": self.collection.attr_collection_endpoint,
                "OPENSEARCH_INDEX": "fate-lore",
                "USE_AWS_AUTH": "true",
                "ENV": env_name,
            },
            tracing=lambda_.Tracing.ACTIVE,
        )

        # ── API Gateway ───────────────────────────────────────────────────────
        self.api = apigwv2.HttpApi(
            self,
            "FateRagApi",
            api_name=f"fate-rag-api-{env_name}",
            cors_preflight=apigwv2.CorsPreflightOptions(
                allow_origins=["*"],
                allow_methods=[apigwv2.CorsHttpMethod.ANY],
                allow_headers=["*"],
            ),
        )

        lambda_integration = integrations.HttpLambdaIntegration(
            "BackendIntegration",
            handler=self.backend_function,
        )

        self.api.add_routes(
            path="/{proxy+}",
            methods=[apigwv2.HttpMethod.ANY],
            integration=lambda_integration,
        )

        # ── Outputs ───────────────────────────────────────────────────────────
        cdk.CfnOutput(
            self,
            "ApiUrl",
            value=self.api.url or "",
            description="API Gateway URL",
            export_name=f"FateRagApiUrl-{env_name}",
        )
        cdk.CfnOutput(
            self,
            "OpenSearchEndpoint",
            value=self.collection.attr_collection_endpoint,
            description="OpenSearch Serverless collection endpoint",
            export_name=f"FateRagOpenSearchEndpoint-{env_name}",
        )
        cdk.CfnOutput(
            self,
            "DocumentsBucketName",
            value=self.documents_bucket.bucket_name,
            description="S3 bucket for raw documents",
            export_name=f"FateRagDocumentsBucket-{env_name}",
        )
