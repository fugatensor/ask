"""Public adapter registry, worker/artifact schemas and registration invariants."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AdapterContractTests(unittest.TestCase):
    def test_registry_and_registration_match_the_public_contract(self):
        registry = json.loads((ROOT / "adapters/registry.json").read_text())
        self.assertEqual(registry["schema_version"], 1)
        self.assertIsInstance(registry["compilers"], dict)
        for compiler_id, directory in registry["compilers"].items():
            with self.subTest(compiler=compiler_id):
                registration = json.loads((ROOT / "adapters" / directory / "registration.json").read_text())
                self.assertEqual(registration["id"], directory)
                self.assertEqual(registration["protocol_version"], 1)
                self.assertIsInstance(registration["supports_fusion"], bool)
                self.assertTrue((ROOT / "adapters" / directory / registration["worker"]).is_file())
                self.assertLessEqual(set(registration.get("aliases", {}).values()), {"python", "worker"})
                for field in ("native_factory",):
                    self.assertIsInstance(registration[field], str)
                self.assertTrue((ROOT / "adapters" / directory / "CMakeLists.txt").is_file())
                if registration["supports_fusion"]:
                    self.assertIsInstance(registration["package"], str)

    def test_public_contract_schemas_are_valid_json_objects(self):
        names = ("execution-plan-v1.schema.json",
                 "external-compiler-request-v1.schema.json",
                 "external-kernel-artifact-v1.schema.json",
                 "adapter-registration-v1.schema.json",
                 "adapter-registry-v1.schema.json")
        for name in names:
            with self.subTest(schema=name):
                value = json.loads((ROOT / "schemas" / name).read_text())
                self.assertEqual(value["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertEqual(value["type"], "object")
                self.assertIn("required", value)


if __name__ == "__main__":
    unittest.main()
