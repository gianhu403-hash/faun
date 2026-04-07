"""Workflow orchestration API router."""

import asyncio

from fastapi import APIRouter
from pydantic import BaseModel

from cloud.workflows.pipeline import get_pipeline_definition
from cloud.workflows.yandex_workflows import register_workflow, run_workflow

router = APIRouter()


@router.get("/api/v1/workflow/definition")
async def workflow_definition():
    """Return the full pipeline definition as JSON."""
    return get_pipeline_definition()


class WorkflowRunRequest(BaseModel):
    scenario: str = "chainsaw"


def _get_run_demo():
    """Lazy import to avoid circular dependency."""
    from cloud.interface.main import _run_demo

    return _run_demo


@router.post("/api/v1/workflow/run")
async def workflow_run(req: WorkflowRunRequest):
    """Run the incident processing pipeline via WorkflowExecutor."""
    reg = await register_workflow()
    result = await run_workflow(reg["workflow_id"], {"scenario": req.scenario})
    run_demo = _get_run_demo()
    asyncio.create_task(run_demo(req.scenario))
    return result
