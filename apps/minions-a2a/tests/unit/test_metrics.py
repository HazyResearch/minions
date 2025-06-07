#!/usr/bin/env python3
"""
Unit tests for A2A-Minions metrics.
Tests metrics collection, tracking, and Prometheus export.
"""

import unittest
import time
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

from a2a_minions.metrics import (
    MetricsManager, request_count, request_duration, task_count,
    task_duration, active_tasks, streaming_sessions, streaming_events,
    auth_attempts, pdf_processed, pdf_processing_duration, client_pool_size,
    stored_tasks, task_evictions, server_info, errors, create_metrics_endpoint,
    registry
)


class TestMetricsManager(unittest.TestCase):
    """Test MetricsManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = MetricsManager()
        
        # Clear all metrics before each test - use registry.unregister and re-register
        # This is a workaround for prometheus_client not having a simple clear method
        # We'll just create new instances for testing
    
    def test_initialization(self):
        """Test metrics manager initialization."""
        # Server info should be set
        # We need to check the actual metrics differently
        metrics_output = registry._collector_to_names
        self.assertIn(server_info, metrics_output)
    
    def test_track_request_success(self):
        """Test tracking successful requests."""
        with self.metrics.track_request("tasks/send"):
            # Simulate some work
            time.sleep(0.01)
        
        # Check request count - we'll check the actual value instead of internal structure
        # Get metrics as text and parse
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_requests_total{method="tasks/send",status="success"}', metrics_text)
    
    def test_track_request_error(self):
        """Test tracking failed requests."""
        try:
            with self.metrics.track_request("tasks/get"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Check metrics were recorded
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_requests_total{method="tasks/get",status="error"}', metrics_text)
        self.assertIn('a2a_minions_errors_total{error_type="ValueError"}', metrics_text)
    
    def test_track_task_success(self):
        """Test tracking successful task execution."""
        # Check initial active tasks
        initial_metrics = self.metrics.get_metrics().decode('utf-8')
        
        with self.metrics.track_task("minion_query"):
            # Active tasks should increase
            during_metrics = self.metrics.get_metrics().decode('utf-8')
            time.sleep(0.01)
        
        # Check task count after completion
        final_metrics = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_tasks_total{skill_id="minion_query",status="completed"}', final_metrics)
    
    def test_track_task_failure(self):
        """Test tracking failed task execution."""
        try:
            with self.metrics.track_task("minions_query"):
                raise RuntimeError("Task failed")
        except RuntimeError:
            pass
        
        # Check task count for failure
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_tasks_total{skill_id="minions_query",status="failed"}', metrics_text)
        self.assertIn('a2a_minions_errors_total{error_type="task_RuntimeError"}', metrics_text)
    
    def test_track_pdf_processing(self):
        """Test tracking PDF processing."""
        with self.metrics.track_pdf_processing():
            time.sleep(0.01)
        
        # Check PDF metrics
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_pdf_processed_total', metrics_text)
        self.assertIn('a2a_minions_pdf_processing_seconds', metrics_text)
    
    def test_track_auth_attempt(self):
        """Test tracking authentication attempts."""
        # Successful auth
        self.metrics.track_auth_attempt("api_key", success=True)
        
        # Failed auth
        self.metrics.track_auth_attempt("api_key", success=False)
        
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_auth_attempts_total{auth_method="api_key",result="success"}', metrics_text)
        self.assertIn('a2a_minions_auth_attempts_total{auth_method="api_key",result="failure"}', metrics_text)
    
    def test_track_streaming_event(self):
        """Test tracking streaming events."""
        initial_metrics = self.metrics.get_metrics().decode('utf-8')
        
        self.metrics.track_streaming_event()
        self.metrics.track_streaming_event()
        
        final_metrics = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_streaming_events_total', final_metrics)
    
    def test_update_streaming_sessions(self):
        """Test updating streaming session count."""
        self.metrics.update_streaming_sessions(5)
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_streaming_sessions 5', metrics_text)
        
        self.metrics.update_streaming_sessions(3)
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_streaming_sessions 3', metrics_text)
    
    def test_update_stored_tasks(self):
        """Test updating stored tasks count."""
        self.metrics.update_stored_tasks(100)
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_stored_tasks 100', metrics_text)
        
        self.metrics.update_stored_tasks(150)
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_stored_tasks 150', metrics_text)
    
    def test_track_task_eviction(self):
        """Test tracking task evictions."""
        self.metrics.track_task_eviction()
        self.metrics.track_task_eviction()
        
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_task_evictions_total', metrics_text)
    
    def test_update_client_pool_stats(self):
        """Test updating client pool statistics."""
        stats = {
            "ollama": 3,
            "openai": 2,
            "anthropic": 1
        }
        
        self.metrics.update_client_pool_stats(stats)
        
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_client_pool_size{provider="ollama"} 3', metrics_text)
        self.assertIn('a2a_minions_client_pool_size{provider="openai"} 2', metrics_text)
        self.assertIn('a2a_minions_client_pool_size{provider="anthropic"} 1', metrics_text)
    
    def test_get_metrics(self):
        """Test getting metrics in Prometheus format."""
        # Generate some metrics
        self.metrics.track_auth_attempt("jwt", True)
        self.metrics.track_streaming_event()
        self.metrics.update_stored_tasks(42)
        
        # Get metrics
        metrics_data = self.metrics.get_metrics()
        
        # Should be bytes
        self.assertIsInstance(metrics_data, bytes)
        
        # Decode and check content
        metrics_text = metrics_data.decode('utf-8')
        
        # Check for expected metrics
        self.assertIn("a2a_minions_auth_attempts_total", metrics_text)
        self.assertIn("a2a_minions_streaming_events_total", metrics_text)
        self.assertIn("a2a_minions_stored_tasks", metrics_text)
        self.assertIn("a2a_minions_server_info", metrics_text)


class TestMetricsEndpoint(unittest.TestCase):
    """Test metrics endpoint creation."""
    
    async def test_create_metrics_endpoint(self):
        """Test creating FastAPI metrics endpoint."""
        # Create endpoint
        endpoint = create_metrics_endpoint()
        
        # Call endpoint
        response = await endpoint()
        
        # Check response
        self.assertEqual(response.media_type, "text/plain; version=0.0.4; charset=utf-8")
        self.assertIsInstance(response.body, bytes)
        
        # Check content includes metrics
        content = response.body.decode('utf-8')
        self.assertIn("a2a_minions", content)


class TestMetricsConcurrency(unittest.TestCase):
    """Test metrics under concurrent access."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = MetricsManager()
    
    def test_concurrent_request_tracking(self):
        """Test concurrent request tracking."""
        import threading
        
        def track_request(method):
            with self.metrics.track_request(method):
                time.sleep(0.01)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            method = f"method_{i % 3}"  # Use 3 different methods
            thread = threading.Thread(target=track_request, args=(method,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check metrics were recorded
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_requests_total', metrics_text)
    
    def test_concurrent_task_tracking(self):
        """Test concurrent task tracking."""
        import threading
        import time
        
        max_concurrent = 0
        max_lock = threading.Lock()
        
        def track_task(skill_id):
            nonlocal max_concurrent
            
            with self.metrics.track_task(skill_id):
                # Just track that we're in a task
                time.sleep(0.02)  # Simulate work
        
        # Create multiple threads
        threads = []
        for i in range(5):
            skill = "minion_query" if i % 2 == 0 else "minions_query"
            thread = threading.Thread(target=track_task, args=(skill,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check metrics
        metrics_text = self.metrics.get_metrics().decode('utf-8')
        self.assertIn('a2a_minions_tasks_total', metrics_text)
        self.assertIn('a2a_minions_active_tasks', metrics_text)


class TestMetricsReset(unittest.TestCase):
    """Test metrics reset functionality."""
    
    def test_metrics_persistence(self):
        """Test that metrics persist across manager instances."""
        # Generate some metrics
        manager1 = MetricsManager()
        manager1.track_auth_attempt("test", True)
        manager1.track_streaming_event()
        
        # Create new manager - metrics should persist
        manager2 = MetricsManager()
        metrics_text = manager2.get_metrics().decode('utf-8')
        
        # Metrics should still be there
        self.assertIn('a2a_minions_streaming_events_total', metrics_text)


if __name__ == "__main__":
    unittest.main()