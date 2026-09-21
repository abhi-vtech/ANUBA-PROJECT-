"""Runtime-neutral analysis core.

Layering, strictly enforced by imports:

    sources/*        know a runtime, know nothing about the rules
    contract.py      the types both sides agree on
    engine.py        the rules, no runtime imports
    order_rules.py   the judgement, no frame concepts
    runner.py        wiring, driven by a pipeline YAML
"""
