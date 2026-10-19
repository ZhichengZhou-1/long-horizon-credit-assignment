"""Best achievable score of a WebShop goal: buy its own target product with the goal's options.

If even the target product scores < 1.0, the goal cannot be completed (success is impossible),
so a failure on it says nothing about which decision was wrong.
Usage: python goal_ceiling.py IDX [IDX ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from webshop_env import make_env  # noqa: E402
from web_agent_site.engine.goal import get_reward  # noqa: E402

env = make_env()
srv = env.server
for idx in map(int, sys.argv[1:]):
    goal = srv.goals[idx]
    asin = goal['asin']
    product = srv.product_item_dict[asin]
    price = srv.product_prices.get(asin)
    reward, info = get_reward(product, goal, price=price, options=goal['goal_options'], verbose=True)
    print(f"idx {idx:3d} ceiling={reward:.3f}  r_type={info.get('r_type')}  r_att={info.get('r_att', 0):.2f}  "
          f"r_option={info.get('r_option')}  r_price={info.get('r_price')}  attrs={goal['attributes']}  "
          f"options={goal['goal_options']}  price_upper={goal['price_upper']} price={price}")
