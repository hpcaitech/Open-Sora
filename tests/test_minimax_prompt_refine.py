"""Unit tests for MiniMax provider support in prompt_refine.py."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opensora.utils.prompt_refine import (
    MINIMAX_BASE_URL,
    MINIMAX_MODELS,
    _get_client,
    _strip_think_tags,
    has_minimax_key,
    refine_prompts_by_minimax,
)


class TestStripThinkTags(unittest.TestCase):
    def test_strips_think_block(self):
        text = "<think>Some reasoning here.</think>\nActual answer."
        self.assertEqual(_strip_think_tags(text), "Actual answer.")

    def test_strips_multiline_think_block(self):
        text = "<think>\nline1\nline2\n</think>\nAnswer."
        self.assertEqual(_strip_think_tags(text), "Answer.")

    def test_no_think_block_unchanged(self):
        text = "Just a plain answer."
        self.assertEqual(_strip_think_tags(text), "Just a plain answer.")

    def test_empty_string(self):
        self.assertEqual(_strip_think_tags(""), "")


class TestMiniMaxConfig(unittest.TestCase):
    def test_minimax_models_list(self):
        self.assertIn("MiniMax-M2.7", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.7-highspeed", MINIMAX_MODELS)
        self.assertEqual(len(MINIMAX_MODELS), 2)

    def test_minimax_base_url_default(self):
        self.assertTrue(MINIMAX_BASE_URL.startswith("https://api.minimax.io"))

    def test_has_minimax_key_false_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            self.assertFalse(has_minimax_key())

    def test_has_minimax_key_true_when_set(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-test-key"}):
            self.assertTrue(has_minimax_key())


class TestGetClient(unittest.TestCase):
    @patch("opensora.utils.prompt_refine.OpenAI")
    def test_minimax_model_uses_minimax_base_url(self, mock_openai):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-test-key"}):
            _get_client("MiniMax-M2.7")
            mock_openai.assert_called_once_with(
                api_key="sk-test-key",
                base_url=MINIMAX_BASE_URL,
            )

    @patch("opensora.utils.prompt_refine.OpenAI")
    def test_minimax_highspeed_uses_minimax_base_url(self, mock_openai):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-test-key"}):
            _get_client("MiniMax-M2.7-highspeed")
            mock_openai.assert_called_once_with(
                api_key="sk-test-key",
                base_url=MINIMAX_BASE_URL,
            )

    @patch("opensora.utils.prompt_refine.OpenAI")
    def test_openai_model_uses_openai_key(self, mock_openai):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-key"}):
            _get_client("gpt-4o")
            mock_openai.assert_called_once_with(api_key="sk-openai-key")

    def test_minimax_model_raises_without_api_key(self):
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            with self.assertRaises(ValueError) as ctx:
                _get_client("MiniMax-M2.7")
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))


class TestRefinePrompt(unittest.TestCase):
    def _make_mock_response(self, content="A refined prompt."):
        choice = MagicMock()
        choice.message.content = content
        response = MagicMock()
        response.choices = [choice]
        return response

    @patch("opensora.utils.prompt_refine.OpenAI")
    def test_refine_prompt_uses_minimax_model(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "A beautiful forest scene with sunlight streaming through the trees."
        )
        mock_openai_cls.return_value = mock_client

        from opensora.utils.prompt_refine import refine_prompt

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-test-key"}):
            result = refine_prompt("a forest", type="t2v", model="MiniMax-M2.7")

        self.assertEqual(result, "A beautiful forest scene with sunlight streaming through the trees.")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")
        self.assertGreater(call_kwargs["temperature"], 0.0)

    @patch("opensora.utils.prompt_refine.OpenAI")
    def test_refine_prompt_default_model_is_gpt4o(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        from opensora.utils.prompt_refine import refine_prompt

        env = {k: v for k, v in os.environ.items() if k != "PROMPT_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("PROMPT_MODEL", None)
            refine_prompt("a forest", type="t2v")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4o")

    @patch("opensora.utils.prompt_refine.OpenAI")
    def test_prompt_model_env_overrides_default(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        from opensora.utils.prompt_refine import refine_prompt

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-key", "PROMPT_MODEL": "MiniMax-M2.7"}):
            refine_prompt("a forest", type="t2v")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")

    @patch("opensora.utils.prompt_refine.OpenAI")
    def test_temperature_is_positive(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        from opensora.utils.prompt_refine import refine_prompt

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-test-key"}):
            refine_prompt("a forest", type="t2v", model="MiniMax-M2.7")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertGreater(call_kwargs["temperature"], 0.0, "temperature must be > 0 for MiniMax")
        self.assertLessEqual(call_kwargs["temperature"], 1.0)

    @patch("opensora.utils.prompt_refine.OpenAI")
    def test_refine_prompts_by_minimax_uses_m27(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response("Refined.")
        mock_openai_cls.return_value = mock_client

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-test-key"}):
            results = refine_prompts_by_minimax(["a dog"], type="t2v")

        self.assertEqual(results, ["Refined."])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")

    @patch("opensora.utils.prompt_refine.OpenAI")
    def test_returns_original_on_failure(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[])
        mock_openai_cls.return_value = mock_client

        from opensora.utils.prompt_refine import refine_prompt

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-test-key"}):
            result = refine_prompt("original prompt", type="t2v", model="MiniMax-M2.7", retry_times=1)

        self.assertEqual(result, "original prompt")


if __name__ == "__main__":
    unittest.main()
