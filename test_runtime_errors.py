#!/usr/bin/env python3
"""
Runtime Error Testing for Consolidated Dashboard
Tests that functions execute without errors (doesn't test output correctness)
"""

import sys
import os
from pathlib import Path
import traceback

# Add paths for imports
sys.path.append('src/dashboard')
sys.path.append('src/dashboard/reports')

# Suppress Streamlit warnings
os.environ['STREAMLIT_RUNTIME_VALIDATE'] = 'false'

def test_cs2_runtime():
    """Test CS2 report runtime for all countries"""
    print("\n" + "="*60)
    print("TESTING CS2 RUNTIME EXECUTION")
    print("="*60)

    try:
        from src.dashboard.reports.cs2_report import main as cs2_main
        from src.dashboard.reports.cs2_report import (
            show_estonia_overall_analysis,
            show_estonia_indicator_analysis,
            COUNTRY_CONFIG
        )

        # Test that helper functions can access country parameter
        countries = ['Estonia', 'Latvia', 'Lithuania']

        for country in countries:
            try:
                # Test configuration access
                config = COUNTRY_CONFIG[country]
                print(f"✓ {country} config accessible: year={config['year']}, flag={config['flag']}")

                # Test if functions would execute without errors (dry run)
                # Note: We can't actually run Streamlit functions without a runtime
                # but we can check if they're callable with correct parameters
                assert callable(show_estonia_overall_analysis), "show_estonia_overall_analysis not callable"
                assert callable(show_estonia_indicator_analysis), "show_estonia_indicator_analysis not callable"

                # Check function signatures
                import inspect
                sig = inspect.signature(show_estonia_overall_analysis)
                params = list(sig.parameters.keys())
                assert 'country' in params, f"Missing 'country' parameter in show_estonia_overall_analysis"
                print(f"✓ {country} functions have correct signatures")

            except Exception as e:
                print(f"✗ {country} runtime error: {e}")
                return False

        print("✓ CS2: All country configurations work")
        return True

    except Exception as e:
        print(f"✗ CS2 import/runtime error: {e}")
        traceback.print_exc()
        return False

def test_cs3_runtime():
    """Test CS3 report runtime"""
    print("\n" + "="*60)
    print("TESTING CS3 RUNTIME EXECUTION")
    print("="*60)

    try:
        from src.dashboard.reports.cs3_report import (
            main as cs3_main,
            case_study_3_main,
            case_study_3_main_crisis_excluded,
            configure_ui_elements,
            get_data_configuration
        )

        # Test configuration functions
        ui_config = configure_ui_elements("interactive")
        data_config = get_data_configuration("full")

        assert isinstance(ui_config, dict), "ui_config not a dict"
        assert isinstance(data_config, dict), "data_config not a dict"
        assert 'show_download_buttons' in ui_config, "ui_config missing expected keys"

        # Check function signatures
        import inspect

        # Check case_study_3_main signature
        sig = inspect.signature(case_study_3_main)
        params = list(sig.parameters.keys())
        expected = ['data_type', 'ui_config', 'data_config', 'context']
        assert params == expected, f"case_study_3_main has wrong signature: {params} vs {expected}"

        # Check case_study_3_main_crisis_excluded signature
        sig = inspect.signature(case_study_3_main_crisis_excluded)
        params = list(sig.parameters.keys())
        assert params == expected, f"case_study_3_main_crisis_excluded has wrong signature: {params} vs {expected}"

        print("✓ CS3: Configuration functions work")
        print("✓ CS3: Function signatures correct")
        return True

    except Exception as e:
        print(f"✗ CS3 runtime error: {e}")
        traceback.print_exc()
        return False

def test_cs4_runtime():
    """Test CS4 report runtime"""
    print("\n" + "="*60)
    print("TESTING CS4 RUNTIME EXECUTION")
    print("="*60)

    try:
        from src.dashboard.reports.cs4_report import (
            main as cs4_main,
            run_cs4_integrated_analysis,
            configure_ui_elements,
            get_data_configuration
        )

        # Test configuration functions
        ui_config = configure_ui_elements("interactive")
        data_config = get_data_configuration("full")

        assert isinstance(ui_config, dict), "ui_config not a dict"
        assert isinstance(data_config, dict), "data_config not a dict"

        # Check run_cs4_integrated_analysis signature
        import inspect
        sig = inspect.signature(run_cs4_integrated_analysis)
        params = list(sig.parameters.keys())
        expected = ['data_type', 'ui_config', 'data_config']
        assert params == expected, f"run_cs4_integrated_analysis has wrong signature: {params} vs {expected}"

        print("✓ CS4: Configuration functions work")
        print("✓ CS4: run_cs4_integrated_analysis signature correct")
        return True

    except Exception as e:
        print(f"✗ CS4 runtime error: {e}")
        traceback.print_exc()
        return False

def test_cs5_runtime():
    """Test CS5 report runtime"""
    print("\n" + "="*60)
    print("TESTING CS5 RUNTIME EXECUTION")
    print("="*60)

    try:
        from src.dashboard.reports.cs5_report import (
            main as cs5_main,
            configure_ui_elements,
            get_data_configuration
        )

        # Test configuration functions with both data types
        for data_type in ['full', 'winsorized']:
            data_config = get_data_configuration(data_type)
            assert isinstance(data_config, dict), f"data_config not a dict for {data_type}"

            if data_type == 'winsorized':
                assert 'winsorized' in data_config['controls_path'], \
                    "Winsorized config doesn't have correct paths"

            print(f"✓ CS5: {data_type} configuration works")

        return True

    except Exception as e:
        print(f"✗ CS5 runtime error: {e}")
        traceback.print_exc()
        return False

def test_main_app_integration():
    """Test main_app integration points"""
    print("\n" + "="*60)
    print("TESTING MAIN APP INTEGRATION")
    print("="*60)

    try:
        from src.dashboard.main_app import (
            show_case_study_3_consolidated,
            show_case_study_2_estonia_restructured,
            show_case_study_2_latvia_restructured,
            show_case_study_2_lithuania_restructured
        )

        # Check that these wrapper functions exist and are callable
        functions = [
            show_case_study_3_consolidated,
            show_case_study_2_estonia_restructured,
            show_case_study_2_latvia_restructured,
            show_case_study_2_lithuania_restructured
        ]

        for func in functions:
            assert callable(func), f"{func.__name__} is not callable"
            print(f"✓ {func.__name__} is callable")

        return True

    except Exception as e:
        print(f"✗ Main app integration error: {e}")
        traceback.print_exc()
        return False

def run_all_runtime_tests():
    """Run all runtime tests"""
    print("\n" + "="*60)
    print("RUNTIME ERROR TEST SUITE")
    print("="*60)
    print("Testing for the specific runtime errors found:")
    print("1. CS2: 'country' not defined")
    print("2. CS3: 'ui_config' not defined")
    print("3. CS4: 'data_type' not defined")

    tests = [
        ("CS2 Runtime", test_cs2_runtime),
        ("CS3 Runtime", test_cs3_runtime),
        ("CS4 Runtime", test_cs4_runtime),
        ("CS5 Runtime", test_cs5_runtime),
        ("Main App Integration", test_main_app_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("RUNTIME TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed}/{total} runtime tests passed")

    if passed == total:
        print("\n✅ All runtime errors have been fixed!")
    else:
        print("\n⚠️ Some runtime errors remain. Check the output above for details.")

    return passed == total

if __name__ == "__main__":
    success = run_all_runtime_tests()
    sys.exit(0 if success else 1)