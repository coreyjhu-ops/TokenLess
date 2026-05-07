"""Chat-first Streamlit UI for the TokenLess prompt mentor."""

from __future__ import annotations

import asyncio
import html
import json
import os
from pathlib import Path
from typing import Any

import requests as _requests
import streamlit as st
import streamlit.components.v1 as components
import yaml
from dotenv import load_dotenv

from src.chat.chatbot import TokenLessChatbot
from src.chat.conversation import ChatSessionState
from src.chat.mentor import TokenLessMentor
from src.core.providers.google_provider import GoogleProvider
from src.core.providers.lmstudio_provider import LMStudioProvider
from src.core.providers.openai_provider import OpenAIProvider
from src.core.token_estimator import TokenEstimator
from src.core.types import OptimizationConstraints
from src.evaluation.judge_pool import JudgePool
from src.evaluation.pairwise_battle import PairwiseBattle
from src.evaluation.self_correction import SelfCorrector
from src.optimization.optimizer import Optimizer
from src.planning.planner import Planner


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "models.yaml"
CHAT_UI_VERSION = "2026-05-07-design-md-ui-refresh"

# ---------------------------------------------------------------------------
# UI Copy
# ---------------------------------------------------------------------------

UI_COPY: dict[str, str] = {
    "kicker": "Vibe Coding Prompt Mentor",
    "subtitle": "Turn vague coding ideas into a high-quality prompt, ready for any AI model.",
    "chat_placeholder": "Describe your project, or continue answering TokenLess questions",
    "spinner_thinking": "TokenLess is processing your answer",
    "sidebar_title": "TokenLess",
    "lmstudio_connected": "LM Studio connected\n`{model_id}`",
    "lmstudio_disconnected": "LM Studio not connected\nStart the local server first",
    "demo_mode_label": "Demo mode",
    "demo_mode_help": "For short prompts, keep the rewrite flow active even if ROI is negative.",
    "new_chat_button": "New chat",
    "runtime_caption": "Runtime",
    "local_model_label": "Local model",
    "token_estimate_header": "Token Estimate",
    "token_estimate_desc": "Estimated tokens consumed when sending the final prompt to the target model: input, reasoning, and reply.",
    "metric_original_tokens": "Original prompt input",
    "metric_optimized_tokens": "Optimized prompt input",
    "metric_reasoning_tokens": "Est. reasoning tokens",
    "metric_answer_tokens": "Est. answer tokens",
    "metric_total_tokens": "Est. total per run",
    "metric_model_score": "Model score",
    "final_prompt_header": "Final Prompt",
    "final_prompt_desc": "This is the optimized prompt you can send directly to your target AI coding model.",
    "copy_button_label": "Copy",
    "copy_button_copied": "Copied",
    "optimized_prompt_label": "Optimized prompt",
    "quality_summary_header": "Quality Summary",
    "understood_task_label": "Understood task",
    "collected_info_label": "Supplementary info included in prompt",
    "optimization_actions_label": "Optimization actions",
    "more_quality_notes": "More quality notes",
    "quality_score_caption": "Quality score breakdown",
    "token_estimate_caption": "Target model token estimate",
}


def t(key: str) -> str:
    """Return UI copy for *key*."""

    return UI_COPY.get(key, key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the chat-first application."""

    st.set_page_config(
        page_title="TokenLess",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_design_system()
    initialize_session_state()
    render_sidebar()

    st.markdown(
        f"""
        <section class="tl-header">
          <p class="tl-kicker">{t("kicker")}</p>
          <h1>TokenLess</h1>
          <p class="tl-subtitle">{t("subtitle")}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    state: ChatSessionState = st.session_state["chat_session"]
    render_chat_history(state)
    if state.stage == "done":
        render_final_result(state)

    user_message = st.chat_input(t("chat_placeholder"))
    if user_message:
        mentor = build_mentor(require_roi=not st.session_state.get("demo_mode", False))
        with st.spinner(t("spinner_thinking")):
            st.session_state["chat_session"] = run_async(
                mentor.receive(state, user_message)
            )
        st.rerun()


def initialize_session_state() -> None:
    """Seed the Streamlit chat session."""

    if "demo_mode" not in st.session_state:
        st.session_state["demo_mode"] = False
    if "chat_session" not in st.session_state:
        mentor = build_mentor(require_roi=not st.session_state["demo_mode"])
        st.session_state["chat_session"] = run_async(mentor.start())


def render_sidebar() -> None:
    """Render concise runtime status and reset controls."""

    with st.sidebar:
        st.header(t("sidebar_title"))
        connected, model_id = _check_lmstudio()
        if connected:
            msg = t("lmstudio_connected").format(model_id=model_id)
            st.success(msg)
        else:
            st.error(t("lmstudio_disconnected"))

        st.checkbox(
            t("demo_mode_label"),
            value=st.session_state.get("demo_mode", False),
            key="demo_mode",
            help=t("demo_mode_help"),
        )
        if st.button(t("new_chat_button"), use_container_width=True):
            mentor = build_mentor(require_roi=not st.session_state["demo_mode"])
            st.session_state["chat_session"] = run_async(mentor.start())
            st.rerun()

        st.divider()
        st.caption(t("runtime_caption"))
        st.write(f"{t('local_model_label')}: `http://localhost:1234/v1`")
        st.write(f"OPENAI_API_KEY: {key_status('OPENAI_API_KEY')}")
        st.write(f"GOOGLE_API_KEY: {key_status('GOOGLE_API_KEY')}")


def render_chat_history(state: ChatSessionState) -> None:
    """Render the mentor conversation."""

    for message in state.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)


def render_final_result(state: ChatSessionState) -> None:
    """Render token estimates, final prompt, and a compact quality summary."""

    mentor = build_mentor(require_roi=not st.session_state.get("demo_mode", False))
    metrics = mentor.user_metrics(state)
    final_prompt = choose_display_prompt(state)

    st.divider()
    render_token_overview(metrics)
    render_final_prompt_module(final_prompt)
    render_user_friendly_details(mentor.advanced_details(state))


def render_token_overview(metrics: dict[str, str]) -> None:
    """Render downstream token estimates as the primary result summary."""

    original_prompt_tokens = _metric_value(metrics, "original_prompt_tokens", "original_tokens")
    optimized_prompt_tokens = _metric_value(metrics, "optimized_prompt_tokens", "optimized_tokens")
    target_reasoning_tokens = _metric_value(metrics, "target_reasoning_tokens")
    target_answer_tokens = _metric_value(metrics, "target_answer_tokens")
    target_total_tokens = _metric_value(metrics, "target_total_tokens")
    model_score = _metric_value(metrics, "model_score")

    st.subheader(t("token_estimate_header"))
    st.markdown(
        f'<p class="tl-muted">{t("token_estimate_desc")}</p>',
        unsafe_allow_html=True,
    )
    prompt_cols = st.columns(2)
    prompt_cols[0].metric(t("metric_original_tokens"), original_prompt_tokens)
    prompt_cols[1].metric(t("metric_optimized_tokens"), optimized_prompt_tokens)

    run_cols = st.columns(4)
    run_cols[0].metric(t("metric_reasoning_tokens"), target_reasoning_tokens)
    run_cols[1].metric(t("metric_answer_tokens"), target_answer_tokens)
    run_cols[2].metric(t("metric_total_tokens"), target_total_tokens)
    run_cols[3].metric(t("metric_model_score"), model_score)


def _metric_value(metrics: dict[str, str], key: str, fallback_key: str | None = None) -> str:
    """Read a metric key with compatibility for active Streamlit sessions."""

    if key in metrics:
        return metrics[key]
    if fallback_key and fallback_key in metrics:
        return metrics[fallback_key]
    return "-"


def choose_display_prompt(state: ChatSessionState) -> str:
    """Choose the best user-facing prompt without relying on cached mentor objects."""

    final_state = state.final_state
    if not final_state:
        return state.raw_prompt

    optimization = final_state.optimization_result
    optimized = optimization.optimizedPrompt if optimization else ""
    pipeline_final = final_state.final_prompt or ""

    if optimized and looks_like_structured_prompt(optimized):
        if not looks_like_structured_prompt(pipeline_final):
            return optimized
        estimator = TokenEstimator()
        if estimator.exact_count(optimized) > estimator.exact_count(pipeline_final) * 1.4:
            return optimized
    return pipeline_final or optimized or state.raw_prompt


def looks_like_structured_prompt(prompt: str) -> bool:
    """Return whether a prompt looks like the optimized Markdown+JSON artifact."""

    return bool(
        prompt
        and "## " in prompt
        and ("```json" in prompt or "Output Format" in prompt)
    )


def render_final_prompt_module(final_prompt: str) -> None:
    """Render final prompt in its own module with a copy button."""

    st.subheader(t("final_prompt_header"))
    st.markdown(
        f'<p class="tl-muted">{t("final_prompt_desc")}</p>',
        unsafe_allow_html=True,
    )
    render_copyable_prompt(final_prompt)


def render_copyable_prompt(final_prompt: str) -> None:
    """Render a readonly prompt block with a browser-side copy button."""

    prompt_json = json.dumps(final_prompt)
    escaped_prompt = html.escape(final_prompt)
    line_count = max(4, final_prompt.count("\n") + 1)
    component_height = min(560, max(220, 96 + line_count * 22))

    copy_label = t("copy_button_label")
    copied_label = t("copy_button_copied")
    optimized_label = t("optimized_prompt_label")

    components.html(
        f"""
        <div class="tl-copy-shell">
          <div class="tl-copy-bar">
            <span>{optimized_label}</span>
            <button id="tl-copy-button" type="button">{copy_label}</button>
          </div>
          <pre id="tl-final-prompt">{escaped_prompt}</pre>
        </div>
        <script>
        const button = document.getElementById("tl-copy-button");
        const promptText = {prompt_json};
        button.addEventListener("click", async () => {{
          await navigator.clipboard.writeText(promptText);
          button.textContent = "{copied_label}";
          setTimeout(() => {{ button.textContent = "{copy_label}"; }}, 1400);
        }});
        </script>
        <style>
        :root {{
          --tl-bg-page: #ffffff;
          --tl-bg-section-warm: #f6f5f4;
          --tl-bg-section-cool: #f5f5f7;
          --tl-bg-card: #ffffff;
          --tl-bg-input: #fafafc;
          --tl-text-primary: #1d1d1f;
          --tl-text-secondary: #615d59;
          --tl-text-muted: #a39e98;
          --tl-text-soft: rgba(0,0,0,0.8);
          --tl-text-tertiary: rgba(0,0,0,0.48);
          --tl-border-default: rgba(0,0,0,0.1);
          --tl-accent-primary: #0071e3;
          --tl-accent-link: #0066cc;
          --tl-badge-info-bg: #f2f9ff;
          --tl-badge-info-text: #097fe8;
          --tl-success: #1aae39;
          --tl-warning: #dd5b00;
          --tl-font-heading: "Anthropic Serif", Georgia, serif;
          --tl-font-body: "Anthropic Sans", Arial, system-ui, sans-serif;
          --tl-font-mono: "Anthropic Mono", Arial, monospace;
          --tl-radius-standard: 8px;
          --tl-radius-comfortable: 11px;
          --tl-shadow-card: rgba(0,0,0,0.04) 0px 4px 18px, rgba(0,0,0,0.027) 0px 2.025px 7.84688px, rgba(0,0,0,0.02) 0px 0.8px 2.925px, rgba(0,0,0,0.01) 0px 0.175px 1.04062px;
        }}
        .tl-copy-shell {{
          border: 1px solid var(--tl-border-default);
          border-radius: var(--tl-radius-standard);
          overflow: hidden;
          background: var(--tl-bg-card);
          box-shadow: var(--tl-shadow-card);
        }}
        .tl-copy-bar {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 16px;
          padding: 12px 16px;
          border-bottom: 1px solid var(--tl-border-default);
          background: var(--tl-bg-section-warm);
          color: var(--tl-text-secondary);
          font: 500 12px/1.25 var(--tl-font-body);
          letter-spacing: 0.12px;
        }}
        #tl-copy-button {{
          min-width: 72px;
          min-height: 32px;
          border: 1px solid transparent;
          border-radius: var(--tl-radius-standard);
          background: var(--tl-accent-primary);
          color: #ffffff;
          cursor: pointer;
          font: 500 16px/1.25 var(--tl-font-body);
        }}
        #tl-copy-button:hover {{
          background: #0780f6;
        }}
        #tl-copy-button:active {{
          transform: scale(0.98);
          background: #0069d2;
        }}
        #tl-copy-button:focus {{
          outline: 2px solid var(--tl-accent-primary);
          outline-offset: 2px;
        }}
        #tl-final-prompt {{
          box-sizing: border-box;
          max-height: 460px;
          margin: 0;
          padding: 16px;
          overflow: auto;
          white-space: pre-wrap;
          color: var(--tl-text-primary);
          background: var(--tl-bg-card);
          font: 400 13px/1.6 var(--tl-font-mono);
        }}
        </style>
        """,
        height=component_height,
    )


def render_user_friendly_details(details: dict[str, object]) -> None:
    """Render useful details without exposing pipeline internals."""

    if not details:
        return

    st.subheader(t("quality_summary_header"))
    summary = details.get("project_summary")
    if summary:
        st.write(f"**{t('understood_task_label')}**:{summary}")

    collected = details.get("collected_details") or {}
    if isinstance(collected, dict) and collected:
        st.write(f"**{t('collected_info_label')}**")
        for key, value in collected.items():
            st.write(f"- {key}: {value}")

    improvements = details.get("applied_improvements") or []
    if improvements:
        readable = [
            str(item).replace("_", " ").replace("professional rewrite", "professional prompt rewrite")
            for item in improvements
        ]
        st.write(f"**{t('optimization_actions_label')}**:" + ", ".join(readable))

    with st.expander(t("more_quality_notes")):
        breakdown = details.get("quality_breakdown")
        usage = details.get("estimated_target_model_usage")
        if breakdown:
            st.caption(t("quality_score_caption"))
            st.json(breakdown)
        if usage:
            st.caption(t("token_estimate_caption"))
            st.json(usage)
        return


def inject_design_system() -> None:
    """Apply the design.md visual system to Streamlit."""

    st.markdown(
        """
        <style>
        :root {
          --tl-bg-page: #ffffff;
          --tl-bg-section-warm: #f6f5f4;
          --tl-bg-section-cool: #f5f5f7;
          --tl-bg-card: #ffffff;
          --tl-bg-card-warm: #f6f5f4;
          --tl-bg-input: #fafafc;
          --tl-text-primary: #1d1d1f;
          --tl-text-secondary: #615d59;
          --tl-text-muted: #a39e98;
          --tl-text-soft: rgba(0,0,0,0.8);
          --tl-text-tertiary: rgba(0,0,0,0.48);
          --tl-border-default: rgba(0,0,0,0.1);
          --tl-border-soft: #dddddd;
          --tl-accent-primary: #0071e3;
          --tl-accent-link: #0066cc;
          --tl-accent-link-dark: #2997ff;
          --tl-badge-info-bg: #f2f9ff;
          --tl-badge-info-text: #097fe8;
          --tl-success: #1aae39;
          --tl-warning: #dd5b00;
          --tl-font-heading: "Anthropic Serif", Georgia, serif;
          --tl-font-body: "Anthropic Sans", Arial, system-ui, sans-serif;
          --tl-font-mono: "Anthropic Mono", Arial, monospace;
          --tl-radius-standard: 8px;
          --tl-radius-comfortable: 11px;
          --tl-radius-large: 12px;
          --tl-radius-pill: 980px;
          --tl-shadow-card: rgba(0,0,0,0.04) 0px 4px 18px, rgba(0,0,0,0.027) 0px 2.025px 7.84688px, rgba(0,0,0,0.02) 0px 0.8px 2.925px, rgba(0,0,0,0.01) 0px 0.175px 1.04062px;
          --tl-shadow-deep: rgba(0,0,0,0.01) 0px 1px 3px, rgba(0,0,0,0.02) 0px 3px 7px, rgba(0,0,0,0.02) 0px 7px 15px, rgba(0,0,0,0.04) 0px 14px 28px, rgba(0,0,0,0.05) 0px 23px 52px;
        }
        html, body, [class*="css"] {
          font-family: var(--tl-font-body);
          color: var(--tl-text-primary);
          letter-spacing: normal;
        }
        .stApp {
          background: var(--tl-bg-page);
        }
        .block-container {
          max-width: 1200px;
          padding-top: 48px;
          padding-bottom: 88px;
        }
        .tl-header {
          max-width: 900px;
          padding: 32px 0 24px;
          margin-bottom: 24px;
          border-bottom: 1px solid var(--tl-border-default);
        }
        .tl-kicker {
          margin: 0 0 8px;
          display: inline-flex;
          align-items: center;
          min-height: 24px;
          padding: 4px 8px;
          border-radius: var(--tl-radius-pill);
          background: var(--tl-badge-info-bg);
          color: var(--tl-badge-info-text);
          font: 400 10px/1.6 var(--tl-font-body);
          letter-spacing: 0.5px;
          text-transform: uppercase;
        }
        .tl-header h1 {
          margin: 0;
          color: var(--tl-text-primary);
          font-family: var(--tl-font-heading);
          font-size: 64px;
          line-height: 1.10;
          font-weight: 500;
          letter-spacing: normal;
        }
        .tl-subtitle {
          margin: 12px 0 0;
          max-width: 720px;
          color: var(--tl-text-secondary);
          font: 400 20px/1.6 var(--tl-font-body);
          letter-spacing: normal;
        }
        .tl-muted {
          margin: 0 0 12px;
          color: var(--tl-text-secondary);
          font: 400 16px/1.6 var(--tl-font-body);
          letter-spacing: normal;
        }
        /* Language toggle button */
        .tl-lang-toggle-wrap {
          display: flex;
          align-items: flex-end;
          justify-content: flex-end;
          padding-top: 40px;
        }
        .tl-lang-toggle-wrap button {
          min-width: 72px !important;
          min-height: 32px !important;
          padding: 4px 14px !important;
          border: 1.5px solid var(--tl-border-default) !important;
          border-radius: var(--tl-radius-pill) !important;
          background: var(--tl-bg-card) !important;
          color: var(--tl-text-secondary) !important;
          font: 500 13px/1.4 var(--tl-font-body) !important;
          box-shadow: none !important;
          letter-spacing: 0.2px !important;
          transition: background 0.15s, color 0.15s, border-color 0.15s;
        }
        .tl-lang-toggle-wrap button:hover {
          border-color: var(--tl-accent-primary) !important;
          color: var(--tl-accent-primary) !important;
          background: var(--tl-badge-info-bg) !important;
        }
        h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
          color: var(--tl-text-primary);
          font-family: var(--tl-font-heading);
          font-weight: 500;
          letter-spacing: normal;
        }
        [data-testid="stMarkdownContainer"] h2 {
          margin-top: 32px;
          font-size: 32px;
          line-height: 1.1;
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
          color: var(--tl-text-secondary);
          font: 400 16px/1.6 var(--tl-font-body);
          letter-spacing: normal;
        }
        [data-testid="stChatMessage"] {
          max-width: 900px;
          border: 1px solid var(--tl-border-default);
          border-radius: var(--tl-radius-standard);
          background: var(--tl-bg-card);
          box-shadow: var(--tl-shadow-card);
          padding: 16px;
          margin-bottom: 16px;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
          background: var(--tl-bg-section-warm);
        }
        div[data-testid="stMetric"] {
          min-height: 116px;
          background: var(--tl-bg-card);
          border: 1px solid var(--tl-border-default);
          border-radius: var(--tl-radius-standard);
          box-shadow: var(--tl-shadow-card);
          padding: 24px;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetricLabel"] {
          color: var(--tl-text-secondary);
          font: 500 12px/1.25 var(--tl-font-body);
          letter-spacing: 0.12px;
        }
        div[data-testid="stMetricValue"] {
          color: var(--tl-text-primary);
          font-family: var(--tl-font-heading);
          font-size: 32px;
          font-weight: 500;
          line-height: 1.1;
          letter-spacing: normal;
        }
        div[data-testid="column"] {
          padding: 0 4px;
        }
        button[kind="primary"], .stButton > button {
          min-height: 40px;
          border: 1px solid transparent;
          border-radius: var(--tl-radius-standard);
          background: rgba(0,0,0,0.05);
          color: var(--tl-text-primary);
          box-shadow: none;
          font: 500 16px/1.25 var(--tl-font-body);
          letter-spacing: normal;
        }
        .stButton > button:hover {
          border-color: transparent;
          background: rgba(0,0,0,0.09);
          color: var(--tl-text-primary);
        }
        .stButton > button:active {
          transform: scale(0.98);
        }
        .stButton > button:focus {
          outline: 2px solid var(--tl-accent-primary);
          outline-offset: 2px;
        }
        textarea, input, [data-baseweb="textarea"] textarea, [data-baseweb="input"] input {
          min-height: 44px;
          border-radius: var(--tl-radius-comfortable) !important;
          background: var(--tl-bg-input) !important;
          color: var(--tl-text-soft) !important;
          font: 400 16px/1.6 var(--tl-font-body) !important;
          letter-spacing: normal !important;
        }
        textarea::placeholder, input::placeholder {
          color: var(--tl-text-tertiary) !important;
        }
        [data-baseweb="textarea"], [data-baseweb="input"] {
          border: 3px solid rgba(0,0,0,0.04) !important;
          border-radius: var(--tl-radius-comfortable) !important;
          background: var(--tl-bg-input) !important;
          box-shadow: none !important;
        }
        [data-baseweb="textarea"]:focus-within,
        [data-baseweb="input"]:focus-within {
          border-color: var(--tl-accent-primary) !important;
        }
        [data-testid="stChatInput"] {
          max-width: 900px;
          margin: 0 auto;
        }
        [data-testid="stSidebar"] {
          background: var(--tl-bg-section-warm);
          border-right: 1px solid var(--tl-border-default);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
          font-family: var(--tl-font-heading);
          font-weight: 500;
          letter-spacing: normal;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
          font-family: var(--tl-font-body);
          letter-spacing: normal;
        }
        [data-testid="stAlert"] {
          border: 1px solid var(--tl-border-default);
          border-radius: var(--tl-radius-standard);
          box-shadow: var(--tl-shadow-card);
        }
        [data-testid="stExpander"] {
          border: 1px solid var(--tl-border-default);
          border-radius: var(--tl-radius-standard);
          box-shadow: var(--tl-shadow-card);
          background: var(--tl-bg-card);
        }
        [data-testid="stExpander"] summary {
          font: 500 16px/1.6 var(--tl-font-body);
          color: var(--tl-text-primary);
        }
        code, pre, .stCode {
          white-space: pre-wrap !important;
          font-family: var(--tl-font-mono) !important;
          letter-spacing: normal !important;
        }
        a {
          color: var(--tl-accent-link);
        }
        @media (max-width: 720px) {
          .block-container {
            padding-top: 16px;
            padding-left: 16px;
            padding-right: 16px;
          }
          .tl-header {
            padding: 24px 0;
          }
          .tl-header h1 {
            font-size: 44px;
          }
          .tl-subtitle {
            font-size: 16px;
          }
          div[data-testid="stMetric"] {
            padding: 16px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def build_mentor(
    require_roi: bool = True,
    cache_version: str = CHAT_UI_VERSION,
) -> TokenLessMentor:
    """Create the mentor wrapper around the reusable pipeline."""

    _ = cache_version
    config = load_model_config()
    token_estimator = TokenEstimator(
        default_encoding=config.get("tokenizer", {}).get("default", "cl100k_base")
    )
    return TokenLessMentor(
        chatbot=build_chatbot(require_roi=require_roi),
        token_estimator=token_estimator,
    )


@st.cache_resource(show_spinner=False)
def build_chatbot(
    require_roi: bool = True,
    cache_version: str = CHAT_UI_VERSION,
) -> TokenLessChatbot:
    """Create and cache the four-layer TokenLess chatbot."""

    _ = cache_version
    load_dotenv(ROOT / ".env")
    config = load_model_config()
    constraints = OptimizationConstraints(
        requirePositiveROI=require_roi,
        maxCompressionRate=0.50,
        minQualityScore=6.0,
        maxSelfCorrectionRetries=2,
    )
    token_estimator = TokenEstimator(
        default_encoding=config.get("tokenizer", {}).get("default", "cl100k_base")
    )

    planning_provider = build_lmstudio_provider(config["planning"])
    optimization_provider = build_lmstudio_provider(config["optimization"])
    evaluation_providers = build_evaluation_providers(config["evaluation"]["judges"])

    planner = Planner(planning_provider)
    optimizer = Optimizer(optimization_provider, token_estimator, constraints)
    judge_pool = JudgePool(evaluation_providers)
    battle = PairwiseBattle(evaluation_providers, judge_pool, constraints)
    corrector = SelfCorrector(optimizer, battle, constraints)
    return TokenLessChatbot(planner, optimizer, battle, corrector)


def load_model_config() -> dict[str, Any]:
    """Read model configuration from config/models.yaml."""

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_lmstudio_provider(config: dict[str, Any]) -> LMStudioProvider:
    """Build a local LM Studio provider from YAML config."""

    return LMStudioProvider(
        model=config["model"],
        base_url=config.get("base_url", "http://localhost:1234/v1"),
    )


def build_evaluation_providers(configs: list[dict[str, Any]]) -> list[Any]:
    """Build hosted judge/target providers from YAML config."""

    providers: list[Any] = []
    for config in configs:
        provider = config["provider"]
        model = config["model"]
        if provider == "openai":
            providers.append(OpenAIProvider(model=model))
        elif provider == "google":
            providers.append(GoogleProvider(model=model))
        else:
            raise ValueError(f"Unsupported evaluation provider: {provider}")
    return providers


def run_async(coro):
    """Run an async call from Streamlit's sync execution model."""

    return asyncio.run(coro)


def key_status(name: str) -> str:
    """Return a concise environment-variable status label."""

    load_dotenv(ROOT / ".env")
    return "set" if os.environ.get(name) else "missing"


def _check_lmstudio() -> tuple[bool, str]:
    """Check whether the LM Studio OpenAI-compatible server is reachable."""

    try:
        response = _requests.get("http://localhost:1234/v1/models", timeout=2)
        if response.status_code == 200:
            models = response.json().get("data", [])
            first = models[0]["id"] if models else "unknown"
            return True, first
        return False, ""
    except Exception:
        return False, ""


if __name__ == "__main__":
    main()
