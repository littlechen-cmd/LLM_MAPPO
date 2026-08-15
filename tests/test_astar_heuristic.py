"""Verify orientation-aware A* heuristic changes."""
from llm_mappo.planner import AStarPlanner
from rware.warehouse import Direction


def test_search_finds_path_with_orientation():
    p = AStarPlanner()
    grid = (10, 10)
    blocked = set()
    # UP -> goal (5,0): needs 1 turn (UP to RIGHT) + 5 forward = 6 steps
    path_up = p._search((0, 0), Direction.UP, (5, 0), grid, blocked)
    assert path_up is not None
    assert path_up[0] == (0, 0)
    assert path_up[-1] == (5, 0)
    # RIGHT -> goal (5,0): already facing right, 5 forward = 5 steps
    path_right = p._search((0, 0), Direction.RIGHT, (5, 0), grid, blocked)
    assert path_right is not None
    assert path_right[-1] == (5, 0)


def test_search_uses_a_stable_tiebreaker_for_equal_priority_states():
    planner = AStarPlanner()
    path = planner._search((4, 4), Direction.UP, (6, 6), (10, 10), set())
    assert path is not None
    assert path[0] == (4, 4)
    assert path[-1] == (6, 6)


def test_heuristic_includes_turn_cost():
    p = AStarPlanner()
    goal = (5, 0)
    # RIGHT is aligned with goal direction → h = 5 + 0 turns = 5
    h_right = p._heuristic((0, 0), Direction.RIGHT, goal)
    assert h_right == 5, f"Expected 5, got {h_right}"
    # UP is 90° off → h = 5 + 1 turn = 6
    h_up = p._heuristic((0, 0), Direction.UP, goal)
    assert h_up == 6, f"Expected 6, got {h_up}"
    # LEFT is 180° off → h = 5 + 2 turns = 7
    h_left = p._heuristic((0, 0), Direction.LEFT, goal)
    assert h_left == 7, f"Expected 7, got {h_left}"


def test_turn_steps():
    p = AStarPlanner()
    assert p._turn_steps(Direction.UP, Direction.UP) == 0
    assert p._turn_steps(Direction.UP, Direction.RIGHT) == 1
    assert p._turn_steps(Direction.UP, Direction.LEFT) == 1
    assert p._turn_steps(Direction.UP, Direction.DOWN) == 2
    assert p._turn_steps(Direction.DOWN, Direction.UP) == 2
    assert p._turn_steps(Direction.LEFT, Direction.RIGHT) == 2


def test_oriented_neighbours_yield_step_costs():
    p = AStarPlanner()
    grid = (10, 10)
    neighbours = list(p._oriented_neighbours((5, 5), Direction.UP, grid))
    # Should yield 4 directional moves + 2 in-place turns = 6 entries
    assert len(neighbours) == 6
    # Find the FORWARD (UP) neighbour: cost should be 1 (no turn)
    up_moves = [(n, d, c) for n, d, c in neighbours if d == Direction.UP]
    assert len(up_moves) == 1
    assert up_moves[0][2] == 1  # step_cost = 1
    # Find the RIGHT neighbour: cost should be 1+1=2 (1 turn + 1 forward)
    right_moves = [
        (n, d, c)
        for n, d, c in neighbours
        if n == (6, 5) and d == Direction.RIGHT
    ]
    assert len(right_moves) == 1
    assert right_moves[0][2] == 2
    # Find the DOWN neighbour: cost should be 1+2=3 (2 turns + 1 forward)
    down_moves = [
        (n, d, c)
        for n, d, c in neighbours
        if n == (5, 6) and d == Direction.DOWN
    ]
    assert len(down_moves) == 1
    assert down_moves[0][2] == 3


def test_reconstruct_oriented_collapses_turns():
    """In-place turns should be collapsed in the reconstructed path."""
    p = AStarPlanner()
    # Simulate: (0,0,UP) -> (0,0,RIGHT) [turn] -> (1,0,RIGHT) [forward]
    came_from = {
        (0, 0, Direction.UP): None,
        (0, 0, Direction.RIGHT): (0, 0, Direction.UP),
        (1, 0, Direction.RIGHT): (0, 0, Direction.RIGHT),
    }
    path = p._reconstruct_oriented(came_from, (1, 0, Direction.RIGHT))
    # Should be [(0,0), (1,0)] — turn collapsed
    assert path == [(0, 0), (1, 0)], f"Expected [(0,0),(1,0)], got {path}"
