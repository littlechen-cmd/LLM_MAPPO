from llm_mappo.behavior_evaluation import (
    AgentState,
    _finalize_group,
    _record_charging_diversion,
)
from llm_mappo.phase2 import Phase2Warehouse
from rware.warehouse import Direction


def test_behavior_group_marks_an_unseen_behavior_as_uncovered():
    assert _finalize_group({"samples": 0, "successes": 0, "charged_events": 0}) == {
        "samples": 0,
        "successes": 0,
        "rate": None,
        "charged_events": 0,
        "covered": False,
    }


def test_low_battery_diversion_counts_progress_toward_the_charger():
    env = Phase2Warehouse(
        n_agents=1,
        max_steps=20,
        env_id="llm-mappo-small-1ag-v1",
        batch_interval=20,
        task_completion_target=3,
    )
    try:
        env.reset(seed=3)
        agent = env.env.agents[0]
        charger = env.env.charging_stations[0]
        agent.x, agent.y = charger[0], charger[1] + 1
        agent.battery = 0.1
        env.env._recalc_grid()
        before = (
            AgentState(
                agent_id=1,
                position=(agent.x, agent.y),
                direction=Direction.DOWN,
                battery=0.1,
                loaded=True,
                priority="A",
                target=charger,
                target_kind="charging",
            ),
        )
        group = {"low_battery_charging_diversion": {"samples": 0, "successes": 0,
                                                     "charged_events": 0}}
        env.env.agents[0].x, env.env.agents[0].y = charger
        env.env._recalc_grid()
        _record_charging_diversion(env, before, {"events": []}, group)
        assert group["low_battery_charging_diversion"]["samples"] == 1
        assert group["low_battery_charging_diversion"]["successes"] == 1
    finally:
        env.close()
