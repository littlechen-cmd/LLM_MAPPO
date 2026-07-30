"""Interactive Phase 1 visualisation for the dynamic LLM-MAPPO warehouse."""

from argparse import ArgumentParser

import gymnasium as gym

import rware  # noqa: F401 - importing registers the Gymnasium environment.
from llm_mappo.planner import AStarPlanner
from rware.warehouse import Action


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--env", default="llm-mappo-medium-3ag-v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser.parse_args()


class WarehouseDemo:
    """Pyglet-driven manual and rule-based execution view for Phase 1."""

    def __init__(self, environment_id, seed, max_steps):
        self.env = gym.make(environment_id, render_mode="human", max_steps=max_steps)
        self.unwrapped = self.env.unwrapped
        self.planner = AStarPlanner()
        self.seed = seed
        self.selected_agent = 0
        self.auto_mode = False
        self._reset()

    def run(self):
        import pyglet

        self.env.render()
        self.unwrapped.renderer.window.on_key_press = self._on_key_press
        interval = 1 / self.unwrapped.metadata["render_fps"]
        pyglet.clock.schedule_interval(self._tick, interval)
        pyglet.app.run()

    def _reset(self):
        _, self.info = self.env.reset(seed=self.seed)
        self.step_count = 0
        self._print_status([0.0] * self.unwrapped.n_agents)

    def _on_key_press(self, key_code, _modifiers):
        from pyglet.window import key

        actions = [Action.NOOP] * self.unwrapped.n_agents
        if key_code == key.TAB:
            self.selected_agent = (self.selected_agent + 1) % self.unwrapped.n_agents
            self._print_status([0.0] * self.unwrapped.n_agents)
            return
        if key_code == key.R:
            self.auto_mode = not self.auto_mode
            print(f"automatic FIFO+A* mode: {'on' if self.auto_mode else 'off'}")
            return
        if key_code == key.N:
            self._reset()
            return
        if key_code == key.ESCAPE:
            self.env.close()
            import pyglet

            pyglet.app.exit()
            return

        manual_action = {
            key.UP: Action.FORWARD,
            key.LEFT: Action.LEFT,
            key.RIGHT: Action.RIGHT,
            key.SPACE: Action.TOGGLE_LOAD,
            key.ENTER: Action.NOOP,
        }.get(key_code)
        if manual_action is not None:
            self.auto_mode = False
            actions[self.selected_agent] = manual_action
            self._step(actions)

    def _tick(self, _elapsed):
        if self.auto_mode:
            self._step(self._automatic_actions())
        self.env.render()

    def _step(self, actions):
        _, rewards, terminated, truncated, self.info = self.env.step(
            [action.value for action in actions]
        )
        self.step_count += 1
        self._print_status(rewards)
        if terminated or truncated:
            print("episode complete; resetting")
            self._reset()

    def _automatic_actions(self):
        actions = []
        for agent in self.unwrapped.agents:
            if agent.dead or agent.picking_lock_steps:
                actions.append(Action.NOOP)
                continue
            task = self.unwrapped.task_queue.task_for_agent(agent.id)
            if task is None:
                actions.append(Action.NOOP)
                continue
            shelf = self.unwrapped.shelfs[task.shelf_id - 1]
            if agent.carrying_shelf:
                target = min(
                    self.unwrapped.goals,
                    key=lambda point: abs(point[0] - agent.x) + abs(point[1] - agent.y),
                )
            else:
                target = shelf.x, shelf.y
            actions.append(self._action_for_target(agent.id, target))
        return actions

    def _action_for_target(self, agent_id, target):
        plan = self.planner.plan(self.unwrapped, agent_id, target)
        if not plan.waypoints:
            return Action.NOOP
        best_action = max(
            range(len(plan.action_preferences)),
            key=plan.action_preferences.__getitem__,
        )
        return Action(best_action)

    def _print_status(self, rewards):
        agents = "; ".join(
            "AGV{agent_id}: battery={battery:.3f}, task={task_id}, lock={lock}".format(
                agent_id=agent["agent_id"],
                battery=agent["battery"],
                task_id=agent["task_id"] or "-",
                lock=agent["picking_lock_steps"],
            )
            for agent in self.info["agents"]
        )
        print(
            f"step={self.info['step']} selected=AGV{self.selected_agent + 1} "
            f"mode={'auto' if self.auto_mode else 'manual'} rewards={rewards} "
            f"queue={self.info['queue']} collisions={self.info['collisions']}"
        )
        print(agents)


if __name__ == "__main__":
    args = parse_args()
    WarehouseDemo(args.env, args.seed, args.max_steps).run()
