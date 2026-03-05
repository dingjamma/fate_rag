"""
Fate RAG CDK Stack
Provisions:
  - S3 bucket for raw documents
  - OpenSearch provisioned domain (t3.small.search, single-node) for vector search
  - Lambda function (FastAPI backend via Mangum)
  - API Gateway HTTP API
  - IAM roles with least-privilege policies
"""

from __future__ import annotations

import os

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_opensearchservice as opensearch
from aws_cdk import aws_s3 as s3
from aws_cdk.aws_lambda_python_alpha import PythonFunction
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
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-*",
                    f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/us.anthropic.claude-*",
                ],
            )
        )

        # S3 read permissions for document bucket
        self.documents_bucket.grant_read(lambda_role)

        # ── OpenSearch provisioned domain ────────────────────────────────────
        # t3.small.search: ~$25/month — predictable cost vs. Serverless OCU billing
        self.domain = opensearch.Domain(
            self,
            "FateLoreDomain",
            domain_name=f"fate-lore-{env_name}",
            version=opensearch.EngineVersion.OPENSEARCH_2_13,
            capacity=opensearch.CapacityConfig(
                data_nodes=1,
                data_node_instance_type="t3.small.search",
            ),
            ebs=opensearch.EbsOptions(
                enabled=True,
                volume_size=10,  # GB — plenty for lore chunks
                volume_type=ec2.EbsDeviceVolumeType.GP3,
            ),
            removal_policy=RemovalPolicy.RETAIN if is_prod else RemovalPolicy.DESTROY,
            access_policies=[
                iam.PolicyStatement(
                    actions=["es:ESHttp*"],
                    principals=[lambda_role],
                    resources=["*"],
                )
            ],
            enforce_https=True,
            node_to_node_encryption=True,
            encryption_at_rest=opensearch.EncryptionAtRestOptions(enabled=True),
        )

        # Scope the Lambda role to just this domain
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="OpenSearchDomainAccess",
                actions=["es:ESHttp*"],
                resources=[f"{self.domain.domain_arn}/*"],
            )
        )

        # ── Lambda function ───────────────────────────────────────────────────
        self.backend_function = PythonFunction(
            self,
            "BackendFunction",
            function_name=f"fate-rag-backend-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_11,
            entry=os.path.join(os.path.dirname(__file__), "../backend"),
            index="app.py",
            handler="handler",
            role=lambda_role,
            timeout=Duration.seconds(60),
            memory_size=512,
            environment={
                "AWS_REGION_NAME": self.region,
                "BEDROCK_REGION": self.region,
                "BEDROCK_MODEL_ID": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "BEDROCK_EMBEDDING_MODEL": "amazon.titan-embed-text-v1",
                "OPENSEARCH_ENDPOINT": f"https://{self.domain.domain_endpoint}",
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
            value=self.domain.domain_endpoint,
            description="OpenSearch provisioned domain endpoint",
            export_name=f"FateRagOpenSearchEndpoint-{env_name}",
        )
        cdk.CfnOutput(
            self,
            "DocumentsBucketName",
            value=self.documents_bucket.bucket_name,
            description="S3 bucket for raw documents",
            export_name=f"FateRagDocumentsBucket-{env_name}",
        )
