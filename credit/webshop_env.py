"""WebShop text environment as used throughout: verl-agent's copy, 1000 products, synthetic goals.

Importing this module puts the WebShop package on sys.path, so `web_agent_site` can be imported
afterwards.
"""
import hashlib
import os
import sys

import config

WEBSHOP_DIR = config.webshop_dir()
DATA_DIR = os.path.join(WEBSHOP_DIR, 'data')
sys.path.insert(0, WEBSHOP_DIR)

GOAL_SEED = 0       # fixes the goal shuffle inside SimServer; session idx -> goal depends on it
MAX_STEPS = 15      # same as verl-agent / HCAPO WebShop runs


def make_env(seed=GOAL_SEED):
    # Built directly rather than via gym.make: gym 0.24's env checker calls reset/step
    # before returning the env, which adds hidden server sessions.
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    return WebAgentTextEnv(
        observation_mode='text',
        num_products=None,
        human_goals=False,
        file_path=os.path.join(DATA_DIR, 'items_shuffle_1000.json'),
        attr_path=os.path.join(DATA_DIR, 'items_ins_v2_1000.json'),
        seed=seed,
    )


def h(text):
    """Short fingerprint of an observation or prompt, used to verify exact state restoration."""
    return hashlib.sha1(text.encode()).hexdigest()[:16]
