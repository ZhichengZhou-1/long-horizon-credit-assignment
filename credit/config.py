"""Paths from environment variables (see env.example.sh).

CREDIT_VERL_AGENT_DIR  checkout of https://github.com/langfengQ/verl-agent (prompt templates and
                       action projection are loaded from it; its vendored WebShop is the env)
CREDIT_WEBSHOP_DIR     optional; defaults to the WebShop copy inside verl-agent
CREDIT_RUNS            where experiment outputs go (used as the default by scripts)
"""
import os


def _require(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'environment variable {name} is not set (see env.example.sh)')
    return value


def verl_agent_dir():
    return _require('CREDIT_VERL_AGENT_DIR')


def webshop_dir():
    return os.environ.get('CREDIT_WEBSHOP_DIR') or os.path.join(
        verl_agent_dir(), 'agent_system', 'environments', 'env_package', 'webshop', 'webshop')


def runs_dir(*parts):
    return os.path.join(_require('CREDIT_RUNS'), *parts)
