#!/usr/bin/env python
"""Demo for Scenario [1-4]"""
import sys
sys.path.insert(0, '..')
from examples.demo_lbcf_mpc import run_scenario_demo, load_pretrained_models

if __name__ == "__main__":
    cbf, gp = load_pretrained_models(Path('../data'))
    run_scenario_demo(scenario=[1/2/3/4], cbf_checkpoint=cbf, gp_model=gp)