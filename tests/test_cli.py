from typer.testing import CliRunner

from leanrank_kg.cli import app


def test_show_neighborhood_uses_entity_id_option():
    runner = CliRunner()
    result = runner.invoke(app, ["show-neighborhood", "--help"])
    assert result.exit_code == 0
    assert "--entity-id" in result.output
