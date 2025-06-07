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
    stored_tasks, task_evictions, server_info, errors, create_metrics_endpoint
)


class TestMetricsManager(unittest.TestCase):
    """Test MetricsManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = MetricsManager()
        
        # Clear all metrics before each test
        request_count._metrics.clear()
        task_count._metrics.clear()
        auth_attempts._metrics.clear()
        errors._metrics.clear()
        pdf_processed._metrics.clear()
        task_evictions._metrics.clear()
        streaming_events._metrics.clear()
    
    def test_initialization(self):
        """Test metrics manager initialization."""
        # Server info should be set
        info_value = list(server_info._metrics.values())[0]
        self.assertEqual(info_value.value['version'], '1.0.0')
        self.assertEqual(info_value.value['protocol'], 'a2a')
        self.assertIn('minion_query', info_value.value['skills'])
    
    def test_track_request_success(self):
        """Test tracking successful requests."""
        with self.metrics.track_request("tasks/send"):
            # Simulate some work
            time.sleep(0.01)
        
        # Check request count
        count_key = ('tasks/send', 'success')
        count_metric = request_count._metrics.get(count_key)
        self.assertIsNotNone(count_metric)
        self.assertEqual(count_metric._value._value, 1)
        
        # Check request duration was recorded
        duration_key = ('tasks/send',)
        duration_metric = request_duration._metrics.get(duration_key)
        self.assertIsNotNone(duration_metric)
        self.assertGreater(duration_metric._sum._value, 0)
    
    def test_track_request_error(self):
        """Test tracking failed requests."""
        try:
            with self.metrics.track_request("tasks/get"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Check request count for error
        count_key = ('tasks/get', 'error')
        count_metric = request_count._metrics.get(count_key)
        self.assertIsNotNone(count_metric)
        self.assertEqual(count_metric._value._value, 1)
        
        # Check error was tracked
        error_key = ('ValueError',)
        error_metric = errors._metrics.get(error_key)
        self.assertIsNotNone(error_metric)
        self.assertEqual(error_metric._value._value, 1)
    
    def test_track_task_success(self):
        """Test tracking successful task execution."""
        # Check initial active tasks
        initial_active = active_tasks._value._value
        
        with self.metrics.track_task("minion_query"):
            # Active tasks should increase
            self.assertEqual(active_tasks._value._value, initial_active + 1)
            time.sleep(0.01)
        
        # Active tasks should decrease after completion
        self.assertEqual(active_tasks._value._value, initial_active)
        
        # Check task count
        count_key = ('minion_query', 'completed')
        count_metric = task_count._metrics.get(count_key)
        self.assertIsNotNone(count_metric)
        self.assertEqual(count_metric._value._value, 1)
        
        # Check task duration
        duration_key = ('minion_query',)
        duration_metric = task_duration._metrics.get(duration_key)
        self.assertIsNotNone(duration_metric)
        self.assertGreater(duration_metric._sum._value, 0)
    
    def test_track_task_failure(self):
        """Test tracking failed task execution."""
        try:
            with self.metrics.track_task("minions_query"):
                raise RuntimeError("Task failed")
        except RuntimeError:
            pass
        
        # Check task count for failure
        count_key = ('minions_query', 'failed')
        count_metric = task_count._metrics.get(count_key)
        self.assertIsNotNone(count_metric)
        self.assertEqual(count_metric._value._value, 1)
        
        # Check error was tracked
        error_key = ('task_RuntimeError',)
        error_metric = errors._metrics.get(error_key)
        self.assertIsNotNone(error_metric)
        self.assertEqual(error_metric._value._value, 1)
    
    def test_track_pdf_processing(self):
        """Test tracking PDF processing."""
        with self.metrics.track_pdf_processing():
            time.sleep(0.01)
        
        # Check PDF count
        self.assertEqual(pdf_processed._value._value, 1)
        
        # Check PDF processing duration
        self.assertGreater(pdf_processing_duration._sum._value, 0)
    
    def test_track_auth_attempt(self):
        """Test tracking authentication attempts."""
        # Successful auth
        self.metrics.track_auth_attempt("api_key", success=True)
        
        success_key = ('api_key', 'success')
        success_metric = auth_attempts._metrics.get(success_key)
        self.assertIsNotNone(success_metric)
        self.assertEqual(success_metric._value._value, 1)
        
        # Failed auth
        self.metrics.track_auth_attempt("api_key", success=False)
        
        failure_key = ('api_key', 'failure')
        failure_metric = auth_attempts._metrics.get(failure_key)
        self.assertIsNotNone(failure_metric)
        self.assertEqual(failure_metric._value._value, 1)
    
    def test_track_streaming_event(self):
        """Test tracking streaming events."""
        initial_count = streaming_events._value._value
        
        self.metrics.track_streaming_event()
        self.assertEqual(streaming_events._value._value, initial_count + 1)
        
        self.metrics.track_streaming_event()
        self.assertEqual(streaming_events._value._value, initial_count + 2)
    
    def test_update_streaming_sessions(self):
        """Test updating streaming session count."""
        self.metrics.update_streaming_sessions(5)
        self.assertEqual(streaming_sessions._value._value, 5)
        
        self.metrics.update_streaming_sessions(3)
        self.assertEqual(streaming_sessions._value._value, 3)
    
    def test_update_stored_tasks(self):
        """Test updating stored tasks count."""
        self.metrics.update_stored_tasks(100)
        self.assertEqual(stored_tasks._value._value, 100)
        
        self.metrics.update_stored_tasks(150)
        self.assertEqual(stored_tasks._value._value, 150)
    
    def test_track_task_eviction(self):
        """Test tracking task evictions."""
        initial_count = task_evictions._value._value
        
        self.metrics.track_task_eviction()
        self.assertEqual(task_evictions._value._value, initial_count + 1)
        
        self.metrics.track_task_eviction()
        self.assertEqual(task_evictions._value._value, initial_count + 2)
    
    def test_update_client_pool_stats(self):
        """Test updating client pool statistics."""
        stats = {
            "ollama": 3,
            "openai": 2,
            "anthropic": 1
        }
        
        self.metrics.update_client_pool_stats(stats)
        
        # Check each provider
        for provider, count in stats.items():
            metric_key = (provider,)
            metric = client_pool_size._metrics.get(metric_key)
            self.assertIsNotNone(metric)
            self.assertEqual(metric._value._value, count)
    
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
        
        # Check total request count
        total_requests = sum(
            metric._value._value 
            for metric in request_count._metrics.values()
        )
        self.assertEqual(total_requests, 10)
    
    def test_concurrent_task_tracking(self):
        """Test concurrent task tracking."""
        import threading
        import time
        
        max_concurrent = 0
        max_lock = threading.Lock()
        
        def track_task(skill_id):
            nonlocal max_concurrent
            
            with self.metrics.track_task(skill_id):
                current_active = active_tasks._value._value
                
                with max_lock:
                    max_concurrent = max(max_concurrent, current_active)
                
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
        
        # Should have tracked concurrent tasks
        self.assertGreater(max_concurrent, 1)
        
        # All tasks should be completed
        self.assertEqual(active_tasks._value._value, 0)


class TestMetricsReset(unittest.TestCase):
    """Test metrics reset functionality."""
    
    def test_clear_metrics(self):
        """Test clearing metrics between tests."""
        # Generate some metrics
        manager = MetricsManager()
        manager.track_auth_attempt("test", True)
        manager.track_streaming_event()
        
        # Clear specific metric
        auth_attempts._metrics.clear()
        
        # Check it's cleared
        self.assertEqual(len(auth_attempts._metrics), 0)
        
        # Other metrics should still exist
        self.assertGreater(streaming_events._value._value, 0)


if __name__ == "__main__":
    unittest.main()