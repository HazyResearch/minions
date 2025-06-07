"""
Metrics collection for A2A-Minions server using Prometheus.
"""

import time
from typing import Optional, Callable
from contextlib import contextmanager
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry
)
from fastapi import Response
import logging

logger = logging.getLogger(__name__)

# Create a custom registry to avoid conflicts
registry = CollectorRegistry()

# Request metrics
request_count = Counter(
    'a2a_minions_requests_total',
    'Total number of requests by method and status',
    ['method', 'status'],
    registry=registry
)

request_duration = Histogram(
    'a2a_minions_request_duration_seconds',
    'Request duration in seconds by method',
    ['method'],
    registry=registry
)

# Task metrics
task_count = Counter(
    'a2a_minions_tasks_total',
    'Total number of tasks by skill and status',
    ['skill_id', 'status'],
    registry=registry
)

task_duration = Histogram(
    'a2a_minions_task_duration_seconds',
    'Task execution duration in seconds by skill',
    ['skill_id'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
    registry=registry
)

active_tasks = Gauge(
    'a2a_minions_active_tasks',
    'Number of currently active tasks',
    registry=registry
)

# Streaming metrics
streaming_sessions = Gauge(
    'a2a_minions_streaming_sessions',
    'Number of active streaming sessions',
    registry=registry
)

streaming_events = Counter(
    'a2a_minions_streaming_events_total',
    'Total number of streaming events sent',
    registry=registry
)

# Authentication metrics
auth_attempts = Counter(
    'a2a_minions_auth_attempts_total',
    'Total authentication attempts by method and result',
    ['auth_method', 'result'],
    registry=registry
)

# PDF processing metrics
pdf_processed = Counter(
    'a2a_minions_pdf_processed_total',
    'Total number of PDFs processed',
    registry=registry
)

pdf_processing_duration = Histogram(
    'a2a_minions_pdf_processing_seconds',
    'PDF processing duration in seconds',
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
    registry=registry
)

# Client pool metrics
client_pool_size = Gauge(
    'a2a_minions_client_pool_size',
    'Number of clients in the connection pool by provider',
    ['provider'],
    registry=registry
)

# Memory management metrics
stored_tasks = Gauge(
    'a2a_minions_stored_tasks',
    'Number of tasks currently stored in memory',
    registry=registry
)

task_evictions = Counter(
    'a2a_minions_task_evictions_total',
    'Total number of tasks evicted from memory',
    registry=registry
)

# Server info
server_info = Info(
    'a2a_minions_server',
    'A2A-Minions server information',
    registry=registry
)

# Error metrics
errors = Counter(
    'a2a_minions_errors_total',
    'Total number of errors by type',
    ['error_type'],
    registry=registry
)


class MetricsManager:
    """Manages metrics collection and reporting."""
    
    def __init__(self):
        # Set initial server info
        server_info.info({
            'version': '1.0.0',
            'protocol': 'a2a',
            'skills': 'minion_query,minions_query'
        })
    
    @contextmanager
    def track_request(self, method: str):
        """Context manager to track request metrics."""
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception as e:
            status = "error"
            errors.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            duration = time.time() - start_time
            request_count.labels(method=method, status=status).inc()
            request_duration.labels(method=method).observe(duration)
    
    @contextmanager
    def track_task(self, skill_id: str):
        """Context manager to track task execution metrics."""
        start_time = time.time()
        active_tasks.inc()
        status = "completed"
        
        try:
            yield
        except Exception as e:
            status = "failed"
            errors.labels(error_type=f"task_{type(e).__name__}").inc()
            raise
        finally:
            duration = time.time() - start_time
            active_tasks.dec()
            task_count.labels(skill_id=skill_id, status=status).inc()
            task_duration.labels(skill_id=skill_id).observe(duration)
    
    @contextmanager
    def track_pdf_processing(self):
        """Context manager to track PDF processing."""
        start_time = time.time()
        
        try:
            yield
            pdf_processed.inc()
        finally:
            duration = time.time() - start_time
            pdf_processing_duration.observe(duration)
    
    def track_auth_attempt(self, auth_method: str, success: bool):
        """Track authentication attempt."""
        result = "success" if success else "failure"
        auth_attempts.labels(auth_method=auth_method, result=result).inc()
    
    def track_streaming_event(self):
        """Track a streaming event."""
        streaming_events.inc()
    
    def update_streaming_sessions(self, count: int):
        """Update active streaming sessions count."""
        streaming_sessions.set(count)
    
    def update_stored_tasks(self, count: int):
        """Update stored tasks count."""
        stored_tasks.set(count)
    
    def track_task_eviction(self):
        """Track a task eviction."""
        task_evictions.inc()
    
    def update_client_pool_stats(self, stats: dict):
        """Update client pool statistics."""
        # Clear existing metrics
        client_pool_size._metrics.clear()
        
        # Update with new stats
        for provider, count in stats.items():
            client_pool_size.labels(provider=provider).set(count)
    
    def get_metrics(self) -> bytes:
        """Generate current metrics in Prometheus format."""
        return generate_latest(registry)


# Global metrics manager instance
metrics_manager = MetricsManager()


def create_metrics_endpoint():
    """Create FastAPI endpoint for metrics."""
    async def metrics_endpoint():
        metrics_data = metrics_manager.get_metrics()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    
    return metrics_endpoint