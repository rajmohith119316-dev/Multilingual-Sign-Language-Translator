"""
tests/run_all_tests.py
Master test runner for the PostgreSQL-to-MongoDB migration & application integrity suite.
Executes all unit and integration test modules and reports individual test results,
execution times, and overall status (PASSED, FAILED, SKIPPED, BLOCKED).
"""

import unittest
import sys
import time
from pathlib import Path

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

# Import test cases
from tests.test_mongo_config_and_connection import TestMongoConfigAndConnection
from tests.test_database_manager_crud import TestDatabaseManagerCRUD
from tests.test_user_manager_crud import TestUserManagerCRUD
from tests.test_migration_script import TestMigrationScript
from tests.test_model_and_inference_pipeline import TestModelAndInferencePipeline
from tests.test_flask_routes import TestFlaskRoutes
from tests.test_cloud_sync_and_bandwidth import TestCloudSyncAndBandwidth
from tests.test_translator import TestTranslator
from tests.test_teach_recording_flow import TestTeachRecordingFlow


class DetailedTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_table = []
        self._test_start = 0

    def startTest(self, test):
        super().startTest(test)
        self._test_start = time.perf_counter()

    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.perf_counter() - self._test_start
        self.results_table.append((test.id(), "PASSED", f"{elapsed:.3f}s", ""))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        elapsed = time.perf_counter() - self._test_start
        self.results_table.append((test.id(), "FAILED", f"{elapsed:.3f}s", str(err[1])))

    def addError(self, test, err):
        super().addError(test, err)
        elapsed = time.perf_counter() - self._test_start
        self.results_table.append((test.id(), "ERROR", f"{elapsed:.3f}s", str(err[1])))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        elapsed = time.perf_counter() - self._test_start
        self.results_table.append((test.id(), "SKIPPED", f"{elapsed:.3f}s", reason))


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestMongoConfigAndConnection,
        TestDatabaseManagerCRUD,
        TestUserManagerCRUD,
        TestMigrationScript,
        TestModelAndInferencePipeline,
        TestFlaskRoutes,
        TestCloudSyncAndBandwidth,
        TestTranslator,
        TestTeachRecordingFlow,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(resultclass=DetailedTestResult, verbosity=1)
    print("=" * 85)
    print("  MULTILINGUAL SIGN LANGUAGE TRANSLATOR - MONGODB ATLAS TEST SUITE")
    print("=" * 85)
    start_total = time.perf_counter()
    result = runner.run(suite)
    total_time = time.perf_counter() - start_total

    print("\n" + "=" * 85)
    print(f"{'TEST NAME':<60} {'STATUS':<10} {'TIME':<10}")
    print("=" * 85)
    for test_id, status, duration, reason in result.results_table:
        # Format shorter name: Module.Class.method
        parts = test_id.split(".")
        short_name = f"{parts[-2]}.{parts[-1]}"
        print(f"{short_name:<60} {status:<10} {duration:<10}")

    print("=" * 85)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Passed: {len(result.results_table) - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print("=" * 85)

    if result.wasSuccessful():
        print(">>> ALL TESTS PASSED SUCCESSFULLY! <<<")
        sys.exit(0)
    else:
        print(">>> SOME TESTS FAILED <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
