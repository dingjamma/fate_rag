#!/usr/bin/env python3
"""
Fate RAG CDK Application Entry Point
Usage:
  cdk deploy --context env=dev
  cdk deploy --context env=prod
"""

import aws_cdk as cdk

from infra.fate_rag_stack import FateRagStack

app = cdk.App()

env_name = app.node.try_get_context("env") or "dev"

FateRagStack(
    app,
    f"FateRag-{env_name.capitalize()}",
    env_name=env_name,
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
    description=f"Fate Series RAG Chatbot infrastructure ({env_name})",
)

app.synth()
