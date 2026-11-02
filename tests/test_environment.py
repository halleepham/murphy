"""Environment smoke tests.

These exist so the Stop hook in .claude/settings.json has something real to
check from the first commit, and so a broken venv is caught immediately rather
than halfway through a build.

They assert only what the assignments mandate. Do not add assertions here for
libraries chosen by a design decision that has not been recorded in
docs/decisions/.
"""

import sys
import unittest


class TestPythonVersion(unittest.TestCase):
    def test_python_is_3_12(self):
        """0003: the project runs on 3.12 (torch 2.2.2 supports up to 3.12)."""
        self.assertEqual(
            sys.version_info[:2], (3, 12), f"got {sys.version.split()[0]}"
        )


class TestMandatedDependencies(unittest.TestCase):
    """Challenge 3 mandates Neo4j. Streamlit was dropped in 0003."""

    def test_neo4j_driver_imports(self):
        import neo4j

        self.assertTrue(neo4j.__version__)

    def test_dotenv_imports(self):
        from dotenv import load_dotenv

        self.assertTrue(callable(load_dotenv))


class TestSecretsAreNotCommitted(unittest.TestCase):
    """A .env must never reach the repository. The old project kept private
    planning docs out of git only by accident; this makes it a test."""

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
