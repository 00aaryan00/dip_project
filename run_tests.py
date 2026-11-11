"""
Script: run_tests.py
Description: Central test runner that executes all unit and integration test suites.
"""

import subprocess
import sys
import os

TEST_FILES = [
    "tests.test_preprocessing",
    "tests.test_quality_analysis",
    "tests.test_sharpness",
    "tests.test_noise",
    "tests.test_segmentation",
    "tests.test_object_analysis",
    "tests.test_feature_extraction",
    "tests.test_baseline",
    "tests.test_classifier",
    "tests.test_end_to_end",
]

def main():
    print("=" * 70)
    print("      RUNNING COMPLETE PROJECT TEST SUITE (10 MODULES)        ")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_mod in TEST_FILES:
        print(f"\n[RUNNING] {test_mod}...")
        res = subprocess.run([sys.executable, "-m", test_mod], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  [PASS] {test_mod}")
            passed += 1
        else:
            print(f"  [FAIL] {test_mod}")
            print(res.stderr or res.stdout)
            failed += 1

    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed} PASSED, {failed} FAILED (Total: {len(TEST_FILES)})")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print("ALL TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

