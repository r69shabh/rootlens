import pytest

from data_engine.scenarios import get_scenario


@pytest.fixture(scope="session")
def healthy_con():
    con, _ = get_scenario("healthy").build_dataset()
    yield con
    con.close()


def make_con(scenario_id: str):
    con, _ = get_scenario(scenario_id).build_dataset()
    return con


@pytest.fixture
def scenario_window():
    from data_engine.generator import DEFAULT_WINDOW_START, WindowConfig

    return WindowConfig(start=DEFAULT_WINDOW_START)
