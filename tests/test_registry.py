from jurassic_gait_studio.registry import bootstrap_workspace, list_bird_clips, list_species


def test_workspace_bootstrap_has_seed_data():
    bootstrap_workspace()
    assert len(list_species("bird")) >= 2
    assert len(list_bird_clips()) >= 2
