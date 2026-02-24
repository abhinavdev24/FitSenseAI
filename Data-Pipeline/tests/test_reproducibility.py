from common.reproducibility import apply_global_seed


def test_apply_global_seed_sets_hash_seed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    apply_global_seed(seed=123, hash_seed="123")

    assert __import__("os").environ["PYTHONHASHSEED"] == "123"
