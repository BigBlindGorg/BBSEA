import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from dependencies import get_rag_engine
from rag.config import settings
from rag.rag_engine_simple import QueenRAGEngine

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthStatus(BaseModel):
    """Health status model."""
    status: str
    timestamp: str
    uptime_seconds: float
    environment: str
    services: dict[str, str]
    system: dict[str, float]


class DetailedHealthStatus(BaseModel):
    """Detailed health status with metrics."""
    status: str
    timestamp: str
    uptime_seconds: float
    environment: str
    services: dict[str, str]
    system: dict[str, Any]
    rag_engine: dict[str, Any]
    storage: dict[str, Any]


# Track startup time
STARTUP_TIME = datetime.now()


@router.get("/", response_model=HealthStatus)
async def health_check(
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> HealthStatus:
    """
    Basic health check endpoint.
    """
    try:
        # Calculate uptime
        uptime = (datetime.now() - STARTUP_TIME).total_seconds()

        # Check RAG engine health
        rag_health = await rag_engine.health_check()

        # Get system metrics
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }

        # Determine overall status
        overall_status = "healthy"
        if rag_health["status"] != "healthy":
            overall_status = "degraded"
        if system_metrics["memory_percent"] > 90 or system_metrics["disk_percent"] > 90:
            overall_status = "warning"

        return HealthStatus(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime,
            environment=settings.environment,
            services={
                "api": "healthy",
                "rag_engine": rag_health["status"],
                "vector_store": "healthy" if rag_health["initialized"] else "initializing"
            },
            system=system_metrics
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}") from e


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check(
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> DetailedHealthStatus:
    """
    Detailed health check with comprehensive metrics.
    """
    try:
        # Calculate uptime
        uptime = (datetime.now() - STARTUP_TIME).total_seconds()

        # Get RAG engine health
        rag_health = await rag_engine.health_check()

        # Get system metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "memory": {
                "total_mb": memory.total / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024 * 1024 * 1024),
                "used_gb": disk.used / (1024 * 1024 * 1024),
                "free_gb": disk.free / (1024 * 1024 * 1024),
                "percent": disk.percent
            },
            "python_version": sys.version,
            "platform": sys.platform
        }

        # Check storage directories
        storage_info = {}
        for dir_name in ["documents", "chroma_db", "logs"]:
            dir_path = Path(f"./storage/{dir_name}")
            if dir_path.exists():
                file_count = len(list(dir_path.iterdir()))
                total_size = sum(f.stat().st_size for f in dir_path.iterdir() if f.is_file())
                storage_info[dir_name] = {
                    "exists": True,
                    "file_count": file_count,
                    "total_size_mb": total_size / (1024 * 1024)
                }
            else:
                storage_info[dir_name] = {
                    "exists": False,
                    "file_count": 0,
                    "total_size_mb": 0
                }

        # Service statuses
        services_status = {
            "api": "healthy",
            "rag_engine": rag_health["status"],
            "vector_store": "healthy" if rag_health["initialized"] else "initializing",
            "openai": "configured" if settings.openai_api_key else "not_configured"
        }

        # Determine overall status
        overall_status = "healthy"
        if rag_health["status"] != "healthy" or not settings.openai_api_key:
            overall_status = "degraded"
        if system_metrics["memory"]["percent"] > 90 or system_metrics["disk"]["percent"] > 90:
            overall_status = "warning"

        return DetailedHealthStatus(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime,
            environment=settings.environment,
            services=services_status,
            system=system_metrics,
            rag_engine=rag_health,
            storage=storage_info
        )

    except Exception as e:
        logger.error(f"Detailed health check error: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}") from e


@router.get("/ready")
async def readiness_check(
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> dict[str, Any]:
    """
    Readiness probe for Kubernetes/Docker deployments.
    Returns 200 if the service is ready to accept requests.
    """
    try:
        # Check if RAG engine is initialized
        rag_health = await rag_engine.health_check()

        if not rag_health["initialized"]:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")

        if not settings.openai_api_key:
            raise HTTPException(status_code=503, detail="OpenAI API key not configured")

        return {
            "ready": True,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check error: {e}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}") from e


@router.get("/live")
async def liveness_check() -> dict[str, Any]:
    """
    Liveness probe for Kubernetes/Docker deployments.
    Returns 200 if the service is alive.
    """
    try:
        return {
            "alive": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Liveness check error: {e}")
        raise HTTPException(status_code=503, detail=f"Service not alive: {str(e)}") from e


@router.get("/startup")
async def startup_check() -> dict[str, Any]:
    """
    Startup probe for Kubernetes/Docker deployments.
    Returns 200 when the service has started successfully.
    """
    try:
        # Check if essential directories exist
        required_dirs = [
            Path("./storage/documents"),
            Path("./storage/chroma_db"),
            Path("./storage/logs")
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                raise HTTPException(
                    status_code=503,
                    detail=f"Required directory {dir_path} does not exist"
                )

        return {
            "started": True,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - STARTUP_TIME).total_seconds()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Startup check error: {e}")
        raise HTTPException(status_code=503, detail=f"Startup failed: {str(e)}") from e
