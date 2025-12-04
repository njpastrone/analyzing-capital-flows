#!/usr/bin/env python3
"""
Comprehensive Test Suite for Consolidated Dashboard
Tests all parameter combinations and critical functionality
"""

import sys
import traceback
from pathlib import Path
import json

# Add paths for imports
sys.path.append('src/dashboard')
sys.path.append('src/dashboard/reports')

def test_imports():
    """Test that all modules import successfully"""
    print("\n" + "="*60)
    print("TESTING MODULE IMPORTS")
    print("="*60)

    results = []
    modules_to_test = [
        ('main_app', 'src.dashboard.main_app'),
        ('cs1_report', 'src.dashboard.reports.cs1_report'),
        ('cs2_report', 'src.dashboard.reports.cs2_report'),
        ('cs3_report', 'src.dashboard.reports.cs3_report'),
        ('cs4_report', 'src.dashboard.reports.cs4_report'),
        ('cs5_report', 'src.dashboard.reports.cs5_report'),
    ]

    for name, module_path in modules_to_test:
        try:
            exec(f'import {module_path}')
            print(f"✓ {name}: Import successful")
            results.append((name, True, None))
        except Exception as e:
            print(f"✗ {name}: {e}")
            results.append((name, False, str(e)))

    return all(r[1] for r in results), results

def test_function_signatures():
    """Test that main functions have correct signatures"""
    print("\n" + "="*60)
    print("TESTING FUNCTION SIGNATURES")
    print("="*60)

    import inspect
    results = []

    test_cases = [
        ('cs1_report', ['data_type', 'output_mode', 'context']),
        ('cs2_report', ['country', 'data_type', 'output_mode', 'context']),
        ('cs3_report', ['data_type', 'output_mode', 'context']),
        ('cs4_report', ['data_type', 'output_mode', 'context']),
        ('cs5_report', ['data_type', 'output_mode', 'context']),
    ]

    for module_name, expected_params in test_cases:
        try:
            module = __import__(f'src.dashboard.reports.{module_name}', fromlist=['main'])
            main_func = getattr(module, 'main')
            sig = inspect.signature(main_func)
            actual_params = list(sig.parameters.keys())

            if actual_params == expected_params:
                print(f"✓ {module_name}: Signature correct {actual_params}")
                results.append((module_name, True, None))
            else:
                error_msg = f"Expected {expected_params}, got {actual_params}"
                print(f"✗ {module_name}: {error_msg}")
                results.append((module_name, False, error_msg))
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            results.append((module_name, False, str(e)))

    return all(r[1] for r in results), results

def test_configuration_functions():
    """Test configuration functions in each module"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION FUNCTIONS")
    print("="*60)

    results = []
    modules = ['cs1_report', 'cs2_report', 'cs3_report', 'cs4_report', 'cs5_report']

    for module_name in modules:
        try:
            module = __import__(f'src.dashboard.reports.{module_name}',
                               fromlist=['get_data_configuration', 'configure_ui_elements'])

            # Test data configuration
            has_data_config = hasattr(module, 'get_data_configuration')
            has_ui_config = hasattr(module, 'configure_ui_elements')

            issues = []
            if not has_data_config:
                issues.append("Missing get_data_configuration")
            if not has_ui_config:
                issues.append("Missing configure_ui_elements")

            if issues:
                print(f"⚠ {module_name}: {', '.join(issues)}")
                results.append((module_name, False, ', '.join(issues)))
            else:
                # Test calling the functions
                try:
                    data_config = module.get_data_configuration('full')
                    ui_config = module.configure_ui_elements('interactive')

                    # Verify return values
                    if not isinstance(data_config, dict):
                        raise ValueError(f"get_data_configuration returned {type(data_config)}, expected dict")
                    if not isinstance(ui_config, dict):
                        raise ValueError(f"configure_ui_elements returned {type(ui_config)}, expected dict")

                    print(f"✓ {module_name}: Configuration functions working")
                    results.append((module_name, True, None))
                except Exception as e:
                    print(f"✗ {module_name}: Error calling config functions: {e}")
                    results.append((module_name, False, str(e)))
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            results.append((module_name, False, str(e)))

    return all(r[1] for r in results), results

def test_parameter_combinations():
    """Test different parameter combinations for each module"""
    print("\n" + "="*60)
    print("TESTING PARAMETER COMBINATIONS")
    print("="*60)

    results = []

    # Define test combinations
    test_combinations = [
        ('cs1_report', [
            {'data_type': 'full', 'output_mode': 'interactive', 'context': 'standalone'},
            {'data_type': 'winsorized', 'output_mode': 'pdf', 'context': 'main_app'},
        ]),
        ('cs2_report', [
            {'country': 'Estonia', 'data_type': 'full', 'output_mode': 'interactive', 'context': 'standalone'},
            {'country': 'Latvia', 'data_type': 'winsorized', 'output_mode': 'pdf', 'context': 'main_app'},
            {'country': 'Lithuania', 'data_type': 'full', 'output_mode': 'interactive', 'context': 'standalone'},
        ]),
        ('cs3_report', [
            {'data_type': 'full', 'output_mode': 'interactive', 'context': 'standalone'},
            {'data_type': 'winsorized', 'output_mode': 'pdf', 'context': 'main_app'},
        ]),
        ('cs4_report', [
            {'data_type': 'full', 'output_mode': 'interactive', 'context': 'standalone'},
            {'data_type': 'winsorized', 'output_mode': 'pdf', 'context': 'main_app'},
        ]),
        ('cs5_report', [
            {'data_type': 'full', 'output_mode': 'interactive', 'context': 'standalone'},
            {'data_type': 'winsorized', 'output_mode': 'pdf', 'context': 'main_app'},
        ]),
    ]

    for module_name, param_sets in test_combinations:
        print(f"\nTesting {module_name}:")
        module_results = []

        for params in param_sets:
            try:
                module = __import__(f'src.dashboard.reports.{module_name}', fromlist=['get_data_configuration', 'configure_ui_elements'])

                # Test configuration functions with parameters
                data_type = params.get('data_type', 'full')
                output_mode = params.get('output_mode', 'interactive')

                data_config = module.get_data_configuration(data_type)
                ui_config = module.configure_ui_elements(output_mode)

                # Verify configurations match expected values
                if data_type == 'winsorized':
                    assert 'winsorized' in str(data_config).lower() or 'outlier' in str(data_config).lower(), \
                        "Winsorized config not detected"

                if output_mode == 'pdf':
                    assert ui_config.get('show_download_buttons') == False, \
                        "PDF mode should disable download buttons"
                    assert ui_config.get('use_expanders') == False, \
                        "PDF mode should disable expanders"

                param_str = ', '.join(f"{k}={v}" for k, v in params.items())
                print(f"  ✓ {param_str}")
                module_results.append(True)

            except Exception as e:
                param_str = ', '.join(f"{k}={v}" for k, v in params.items())
                print(f"  ✗ {param_str}: {e}")
                module_results.append(False)

        results.append((module_name, all(module_results), None))

    return all(r[1] for r in results), results

def test_data_paths():
    """Test that data paths are correctly configured"""
    print("\n" + "="*60)
    print("TESTING DATA PATHS")
    print("="*60)

    from pathlib import Path
    results = []

    # Check critical data directories
    base_path = Path(__file__).parent / "updated_data" / "Clean"

    critical_paths = [
        (base_path / "comprehensive_df_PGDP_labeled.csv", "Main dataset"),
        (base_path / "comprehensive_df_PGDP_labeled_winsorized.csv", "Winsorized dataset"),
        (base_path / "CS4_Statistical_Modeling", "CS4 full data"),
        (base_path / "CS4_Statistical_Modeling_winsorized", "CS4 winsorized data"),
        (base_path / "CS5_Capital_Controls", "CS5 capital controls"),
        (base_path / "CS5_Regime_Analysis", "CS5 regime analysis"),
    ]

    for path, description in critical_paths:
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✓ {description}: Found ({size_mb:.2f} MB)")
            else:
                print(f"✓ {description}: Directory exists")
            results.append((description, True, None))
        else:
            print(f"✗ {description}: Not found at {path}")
            results.append((description, False, f"Path not found: {path}"))

    return all(r[1] for r in results), results

def run_all_tests():
    """Run all tests and generate summary report"""
    print("\n" + "="*60)
    print("CONSOLIDATED DASHBOARD TEST SUITE")
    print("="*60)

    test_suites = [
        ("Module Imports", test_imports),
        ("Function Signatures", test_function_signatures),
        ("Configuration Functions", test_configuration_functions),
        ("Parameter Combinations", test_parameter_combinations),
        ("Data Paths", test_data_paths),
    ]

    all_results = {}

    for suite_name, test_func in test_suites:
        try:
            success, results = test_func()
            all_results[suite_name] = {
                'success': success,
                'results': results
            }
        except Exception as e:
            print(f"\n✗ {suite_name} suite failed: {e}")
            traceback.print_exc()
            all_results[suite_name] = {
                'success': False,
                'results': [],
                'error': str(e)
            }

    # Generate summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_suites = len(test_suites)
    passed_suites = sum(1 for r in all_results.values() if r['success'])

    for suite_name, result in all_results.items():
        status = "✓ PASSED" if result['success'] else "✗ FAILED"
        print(f"{status}: {suite_name}")

        if not result['success'] and 'results' in result:
            failed_tests = [r for r in result['results'] if r and not r[1]]
            for test_name, _, error in failed_tests[:3]:  # Show first 3 failures
                print(f"  - {test_name}: {error}")

    print(f"\nOverall: {passed_suites}/{total_suites} test suites passed")

    # Save detailed results
    results_file = Path(__file__).parent / "test_results.json"
    with open(results_file, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for suite, data in all_results.items():
            json_results[suite] = {
                'success': data['success'],
                'details': str(data.get('results', [])),
                'error': data.get('error', None)
            }
        json.dump(json_results, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    return passed_suites == total_suites

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)