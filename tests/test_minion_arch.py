import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.minion_arch import Minion
from minions.clients.openai import OpenAIClient
from minions.clients.ollama import OllamaClient
from minions.utils.arch_router import ArchRouter
from minions.usage import Usage


class TestMinionArch(unittest.TestCase):
    """Test Arch Router Minion protocol integration."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            self.skipTest("OPENAI_API_KEY not set - skipping Arch Router tests")

        self.local_clients = {
            "llama3.2": OllamaClient(model_name="llama3.2", temperature=0.0),
        }

        self.client_metadata = {
            "llama3.2": {
                "name": "llama3.2",
                "description": "General purpose Q&A and reasoning for general domain tasks. Low cost."
            },
        }

        self.remote_client = OpenAIClient(
            model_name="gpt-4o-mini",
            temperature=0.0
        )

    def test_minion_arch_initialization(self):
        """Test that Arch Router Minion initializes correctly."""
        minion = Minion(
            remote_client=self.remote_client,
            local_clients=self.local_clients,
            client_metadata=self.client_metadata,
            max_rounds=1
        )

        self.assertIsNotNone(minion)
        self.assertEqual(len(minion.local_clients), 1)
        self.assertIn("llama3.2", minion.local_clients)

    def test_minion_arch_empty_clients_raises_error(self):
        """Test that initializing with empty local_clients raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Minion(
                remote_client=self.remote_client,
                local_clients={},
                max_rounds=1
            )

        self.assertIn("local_clients cannot be empty", str(context.exception))

    def test_minion_arch_basic_execution(self):
        """Test basic Arch Router Minion protocol execution.

        WARNING: This test loads Arch Router model and performs inference.
        Can take 15+ minutes on CPU. Set RUN_SLOW_TESTS=1 to enable.
        """
        if not os.getenv("RUN_SLOW_TESTS"):
            self.skipTest("Skipping slow test. Set RUN_SLOW_TESTS=1 to enable (requires GPU or 3-4 hours on CPU)")

        minion = Minion(
            remote_client=self.remote_client,
            local_clients=self.local_clients,
            client_metadata=self.client_metadata,
            max_rounds=2
        )

        context = [
            "The capital of France is Paris. Paris is known for the Eiffel Tower."
        ]

        result = minion(
            task="What is the capital of France?",
            context=context
        )

        self.assertIn("final_answer", result)
        self.assertIn("selected_client", result)
        self.assertIn("routing_decision", result)
        self.assertIn("remote_usage", result)
        self.assertIn("local_usage", result)
        self.assertIn("log_file", result)

        self.assertIsInstance(result["routing_decision"], dict)
        self.assertIn("route", result["routing_decision"])

        self.assertIn(result["selected_client"], self.local_clients.keys())

        self.assertIsInstance(result["remote_usage"], Usage)
        self.assertIsInstance(result["local_usage"], Usage)
        self.assertGreater(result["remote_usage"].total_tokens, 0)
        self.assertGreater(result["local_usage"].total_tokens, 0)

        final_answer = result["final_answer"].lower()
        self.assertIn("paris", final_answer)

    def test_minion_arch_default_metadata(self):
        """Test that default metadata is generated when not provided."""
        minion = Minion(
            remote_client=self.remote_client,
            local_clients=self.local_clients,
            max_rounds=1
        )

        self.assertIsNotNone(minion.client_metadata)
        self.assertIn("llama3.2", minion.client_metadata)
        self.assertIn("name", minion.client_metadata["llama3.2"])
        self.assertIn("description", minion.client_metadata["llama3.2"])

    def test_minion_arch_with_custom_router(self):
        """Test using a pre-initialized ArchRouter instance."""
        custom_router = ArchRouter(verbose=False)

        minion = Minion(
            remote_client=self.remote_client,
            local_clients=self.local_clients,
            client_metadata=self.client_metadata,
            arch_router=custom_router,
            max_rounds=1
        )

        self.assertIs(minion._arch_router, custom_router)


class TestArchRouterUtility(unittest.TestCase):
    """Test standalone ArchRouter utility functionality."""

    def test_arch_router_prompt_format(self):
        """Test that ArchRouter generates correct XML-tagged prompt format."""
        try:
            # Mock the transformer components to avoid loading the model
            with patch('minions.utils.arch_router.AutoModelForCausalLM') as mock_model_cls, \
                 patch('minions.utils.arch_router.AutoTokenizer') as mock_tokenizer_cls, \
                 patch('minions.utils.arch_router.torch') as mock_torch:

                # Setup mocks
                mock_torch.cuda.is_available.return_value = False
                mock_tokenizer = MagicMock()
                mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

                mock_model = MagicMock()
                mock_model_cls.from_pretrained.return_value = mock_model

                # Mock tokenizer to capture the prompt
                captured_prompt = None
                def capture_prompt(prompt, **_):
                    nonlocal captured_prompt
                    captured_prompt = prompt
                    mock_inputs = MagicMock()
                    mock_inputs.to.return_value = mock_inputs
                    return mock_inputs

                mock_tokenizer.side_effect = capture_prompt
                mock_tokenizer.eos_token_id = 0

                # Mock model generate to return a JSON response
                mock_output = MagicMock()
                mock_model.generate.return_value = mock_output
                mock_tokenizer.decode.return_value = '{"route": "general"}'

                # Create router and test routing
                router = ArchRouter(verbose=False)

                available_clients = {
                    "general": {
                        "name": "general",
                        "description": "General purpose Q&A tasks"
                    },
                    "code": {
                        "name": "code",
                        "description": "Code debugging tasks"
                    }
                }

                router.route(
                    query="What is the capital of France?",
                    available_clients=available_clients
                )

                # Validate the prompt format
                self.assertIsNotNone(captured_prompt, "Prompt should have been captured")

                # Check for required XML tags (main validation)
                self.assertIn("<routes>", captured_prompt, "Prompt should contain <routes> tag")
                self.assertIn("</routes>", captured_prompt, "Prompt should contain </routes> tag")
                self.assertIn("<conversation>", captured_prompt, "Prompt should contain <conversation> tag")
                self.assertIn("</conversation>", captured_prompt, "Prompt should contain </conversation> tag")

                # Check that routes section contains JSON array structure
                self.assertIn('"name":', captured_prompt, "Routes should contain 'name' field")
                self.assertIn('"description":', captured_prompt, "Routes should contain 'description' field")
                self.assertIn('"general"', captured_prompt, "Should contain 'general' route")
                self.assertIn('"code"', captured_prompt, "Should contain 'code' route")

                # Check conversation section contains the query
                self.assertIn('"role": "user"', captured_prompt, "Conversation should have user role")
                self.assertIn('"content": "What is the capital of France?"', captured_prompt, "Conversation should contain query")

                # Check for task instruction
                self.assertIn("You are a helpful assistant designed to find the best suited route", captured_prompt)
                self.assertIn("Your task is to decide which route is best suit", captured_prompt)

        except ImportError as e:
            self.skipTest(f"transformers not available: {e}")

    def test_arch_router_initialization(self):
        """Test that ArchRouter initializes correctly.

        WARNING: This test loads a 1.5B parameter model and can take 2-3 minutes on CPU.
        Set RUN_SLOW_TESTS=1 environment variable to enable this test.
        """
        if not os.getenv("RUN_SLOW_TESTS"):
            self.skipTest("Skipping slow test. Set RUN_SLOW_TESTS=1 to enable (requires GPU or 3-4 hours on CPU)")

        try:
            router = ArchRouter(verbose=False)
            self.assertIsNotNone(router)
            self.assertIsNotNone(router.model)
            self.assertIsNotNone(router.tokenizer)
        except ImportError as e:
            self.skipTest(f"transformers not available: {e}")

    def test_arch_router_basic_routing(self):
        """Test basic routing decision with ArchRouter.

        WARNING: This test performs actual model inference and can take 10+ minutes on CPU.
        Set RUN_SLOW_TESTS=1 environment variable to enable this test.
        """
        if not os.getenv("RUN_SLOW_TESTS"):
            self.skipTest("Skipping slow test. Set RUN_SLOW_TESTS=1 to enable (requires GPU or 3-4 hours on CPU)")

        try:
            router = ArchRouter(verbose=False)

            available_clients = {
                "general": {
                    "name": "general",
                    "description": "General purpose Q&A and reasoning tasks"
                },
                "code": {
                    "name": "code",
                    "description": "Code debugging and generation tasks"
                }
            }

            decision = router.route(
                query="What is the capital of France?",
                available_clients=available_clients
            )

            self.assertIsInstance(decision, dict)
            self.assertIn("route", decision)
            self.assertIsInstance(decision["route"], str)
            self.assertIn(decision["route"], available_clients.keys())

        except ImportError as e:
            self.skipTest(f"transformers not available: {e}")

    def test_arch_router_code_task_routing(self):
        """Test that ArchRouter correctly routes code-related tasks.

        WARNING: This test performs actual model inference and can take 10+ minutes on CPU.
        Set RUN_SLOW_TESTS=1 environment variable to enable this test.
        """
        if not os.getenv("RUN_SLOW_TESTS"):
            self.skipTest("Skipping slow test. Set RUN_SLOW_TESTS=1 to enable (requires GPU or 3-4 hours on CPU)")

        try:
            router = ArchRouter(verbose=False)

            available_clients = {
                "llama3.2": {
                    "name": "llama3.2",
                    "description": "General purpose Q&A for general domain tasks"
                },
                "deepseek-coder": {
                    "name": "deepseek-coder",
                    "description": "Code debugging and generation for code domain tasks"
                }
            }

            decision = router.route(
                query="Fix this Python bug: NameError on line 10",
                available_clients=available_clients
            )

            self.assertIn("route", decision)
            self.assertIn(decision["route"], available_clients.keys())

        except ImportError as e:
            self.skipTest(f"transformers not available: {e}")

    def test_arch_router_empty_clients_raises_error(self):
        """Test that routing with empty clients raises ValueError."""
        try:
            router = ArchRouter(verbose=False)

            with self.assertRaises(ValueError) as context:
                router.route(
                    query="Test query",
                    available_clients={}
                )

            self.assertIn("available_clients cannot be empty", str(context.exception))

        except ImportError as e:
            self.skipTest(f"transformers not available: {e}")

    def test_arch_router_empty_query_raises_error(self):
        """Test that routing with empty query raises ValueError."""
        try:
            router = ArchRouter(verbose=False)

            available_clients = {
                "test": {"capabilities": "Test model"}
            }

            with self.assertRaises(ValueError) as context:
                router.route(
                    query="",
                    available_clients=available_clients
                )

            self.assertIn("query cannot be empty", str(context.exception))

        except ImportError as e:
            self.skipTest(f"transformers not available: {e}")

    def test_arch_router_batch_routing(self):
        """Test batch routing of multiple queries.

        WARNING: This test performs multiple model inferences and can take 20+ minutes on CPU.
        Set RUN_SLOW_TESTS=1 environment variable to enable this test.
        """
        if not os.getenv("RUN_SLOW_TESTS"):
            self.skipTest("Skipping slow test. Set RUN_SLOW_TESTS=1 to enable (requires GPU or 3-4 hours on CPU)")

        try:
            router = ArchRouter(verbose=False)

            available_clients = {
                "general": {
                    "name": "general",
                    "description": "General purpose Q&A tasks"
                },
                "code": {
                    "name": "code",
                    "description": "Code debugging tasks"
                }
            }

            queries = [
                "What is the capital of France?",
                "Fix this Python error"
            ]

            decisions = router.batch_route(
                queries=queries,
                available_clients=available_clients
            )

            self.assertEqual(len(decisions), len(queries))
            for decision in decisions:
                self.assertIn("route", decision)
                self.assertIn(decision["route"], available_clients.keys())

        except ImportError as e:
            self.skipTest(f"transformers not available: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
