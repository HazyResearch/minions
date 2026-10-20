"""Exercise the reusable Deep Research UI with Streamlit's real widget runner."""

import os
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP = '''
import streamlit as st
from minions.utils.app_utils import render_deep_research_ui

class ResearchProbe:
    search_provider = st.session_state.get("instance_provider", "auto")
    def __call__(self, **kwargs):
        st.session_state["research_call"] = kwargs
        return "Research result", ["https://example.com/source"]

render_deep_research_ui(ResearchProbe())
'''


class ParallelDeepResearchUITests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_no_key_user_can_start_research(self):
        app = AppTest.from_string(APP).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        app.text_area[0].input("Minions research query")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["research_call"]["search_provider"], "auto")
        self.assertTrue(any("Research result" in item.value for item in app.markdown))

    def test_saved_selection_disabled_and_explicit_override(self):
        app = AppTest.from_string(APP).run()
        app.selectbox[0].select("disabled").run()
        app.text_area[0].input("Offline research query")
        app.button[0].click().run()
        self.assertEqual(app.session_state["research_call"]["search_provider"], "disabled")
        self.assertEqual(app.selectbox[0].value, "disabled")
        app.selectbox[0].select("parallel").run()
        app.text_input[0].input("configured-firecrawl-key").run()
        app.button[0].click().run()
        self.assertFalse(app.error)
        self.assertEqual(app.session_state["research_call"]["search_provider"], "parallel")

    def test_partial_legacy_configuration_requires_completion(self):
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "configured-key"}):
            app = AppTest.from_string(APP).run()
            app.text_area[0].input("Configured search query")
            app.button[0].click().run()
        self.assertTrue(app.error)
        self.assertIn("requires both", app.error[0].value)
        self.assertFalse(app.exception)

    def test_instance_provider_is_the_initial_ui_choice(self):
        app = AppTest.from_string(APP)
        app.session_state["instance_provider"] = "disabled"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value, "disabled")


if __name__ == "__main__":
    unittest.main()
