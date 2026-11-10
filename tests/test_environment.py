"""Environment smoke tests.

A broken environment is caught here, before a long build fails halfway.
"""

import sys
import unittest


class TestPythonVersion(unittest.TestCase):
    def test_python_is_3_12(self):
        """The project runs on 3.12 (torch 2.2.2 supports up to 3.12)."""
        self.assertEqual(
            sys.version_info[:2], (3, 12), f"got {sys.version.split()[0]}"
        )


class TestMandatedDependencies(unittest.TestCase):
    """The knowledge graph needs the Neo4j driver."""

    def test_neo4j_driver_imports(self):
        import neo4j

        self.assertTrue(neo4j.__version__)

    def test_dotenv_imports(self):
        from dotenv import load_dotenv

        self.assertTrue(callable(load_dotenv))


class TestBuildStack(unittest.TestCase):
    """The approved stack only works as a set on this Intel Mac:
    torch 2.2.2 is the last macOS x86_64 wheel and needs numpy<2, which in
    turn caps pyarrow at 25. Fail loudly if an upgrade breaks that."""

    def test_numpy_below_2(self):
        import numpy

        self.assertLess(int(numpy.__version__.split(".")[0]), 2, numpy.__version__)

    def test_stack_imports_together(self):
        import duckdb, pandas, pyarrow, sklearn, torch, torch_geometric  # noqa: F401
        import sentence_transformers, rank_bm25, networkx, fastapi  # noqa: F401

        self.assertEqual(torch.tensor([1.0]).numpy().tolist(), [1.0])


class TestSecretsAreNotCommitted(unittest.TestCase):
    """A .env must never reach the repository."""

    def test_env_is_gitignored(self):
        import pathlib
        import subprocess

        root = pathlib.Path(__file__).resolve().parent.parent
        if not (root / ".git").exists():
            self.skipTest("not a git checkout")

        result = subprocess.run(
            ["git", "check-ignore", "-q", ".env"], cwd=root, check=False
        )
        self.assertEqual(result.returncode, 0, ".env is NOT gitignored")

    def test_private_dir_is_gitignored(self):
        import pathlib
        import subprocess

        root = pathlib.Path(__file__).resolve().parent.parent
        if not (root / ".git").exists():
            self.skipTest("not a git checkout")

        result = subprocess.run(
            ["git", "check-ignore", "-q", "private/"], cwd=root, check=False
        )
        self.assertEqual(result.returncode, 0, "private/ is NOT gitignored")


if __name__ == "__main__":
    unittest.main()
