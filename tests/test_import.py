"""Test that core modules can be imported."""


def test_import_eldengym():
    """Test that eldengym package can be imported."""
    import eldengym

    assert eldengym is not None


def test_import_env():
    """Test that EldenGymEnv can be imported."""
    from eldengym.env import EldenGymEnv

    assert EldenGymEnv is not None


def test_import_siphon_client():
    """Test that SiphonClient can be imported."""
    from pysiphon import SiphonClient

    assert SiphonClient is not None


def test_import_elden_client():
    """Test that EldenClient can be imported."""
    from eldengym.client.elden_client import EldenClient

    assert EldenClient is not None


def test_import_utils():
    """Test that utils module can be imported."""
    from eldengym import utils

    assert utils is not None
    assert hasattr(utils, "resolve_file_path")


def test_import_rewards():
    """Test that rewards module can be imported."""
    from eldengym import rewards

    assert rewards is not None
    assert hasattr(rewards, "RewardFunction")
    assert hasattr(rewards, "ScoreDeltaReward")
    assert hasattr(rewards, "BossDefeatReward")
    assert hasattr(rewards, "CustomReward")
