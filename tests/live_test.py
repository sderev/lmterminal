import pytest
from click.testing import CliRunner

from lmterminal import cli


@pytest.mark.integration
def test_live_call(live_model):
    result = CliRunner().invoke(cli.lmt, ["--model", live_model, "Ping"])
    assert result.exit_code == 0, result.output
