"""Merge runs on disjoint goal ranges that share one layout (e.g. goals 0-99 and the two halves of
goals 100-599) into one directory, so the analysis scripts can read them as a single run.

Concatenates plan.json / q_table.json (lists) and *.jsonl / goal ceiling files under each of the
given sub-paths. Every input root must have the same sub-paths.

Usage: python3 merge_runs.py OUT ROOT1 ROOT2 [...]
"""
import json
import os
import sys

SUBDIRS = ['gigpo_eval/gigpo7b_t1.0', 'branch_full', 'reeval16', 'hcapo_action', 'hcapo_response']
LISTS = ('plan.json', 'q_table.json')
LINES = ('episodes.jsonl', 'branches.jsonl', 'hcapo_scores.jsonl')


def main(out, roots):
    for sub in SUBDIRS:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
        for name in LISTS + LINES:
            paths = [os.path.join(r, sub, name) for r in roots]
            have = [p for p in paths if os.path.exists(p)]
            if not have:
                continue
            if len(have) != len(paths):
                sys.exit(f'{sub}/{name} missing in: {[p for p in paths if p not in have]}')
            dst = os.path.join(out, sub, name)
            if name in LISTS:
                json.dump([x for p in have for x in json.load(open(p))], open(dst, 'w'), indent=1)
            else:
                with open(dst, 'w') as f:
                    for p in have:
                        f.writelines(open(p))
            print(f'{dst}: from {len(have)} runs')
    with open(os.path.join(out, 'goal_ceiling.txt'), 'w') as f:
        for r in roots:
            f.writelines(l for l in open(os.path.join(r, 'goal_ceiling.txt')) if l.startswith('idx'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
