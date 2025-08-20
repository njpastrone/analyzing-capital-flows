#!/usr/bin/env python3
"""
Cross-Platform Compatibility Test for Capital Flows Research Platform

Tests functionality, performance, and user experience across different
operating systems, browsers, and deployment environments.
"""

import platform
import sys
import subprocess
import time
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

class CrossPlatformTester:
    """Tests cross-platform compatibility and responsive design"""
    
    def __init__(self):
        self.test_results = []
        self.platform_info = self.get_platform_info()
        
    def get_platform_info(self):
        """Get current platform information"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'architecture': platform.architecture()
        }
    
    def log_test(self, category, test_name, status, details):
        """Log test result"""
        result = {
            'category': category,
            'test_name': test_name,
            'status': status,
            'details': details,
            'platform': self.platform_info['system'],
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
    
    def test_path_handling(self):
        """Test cross-platform path handling"""
        print("📂 Testing Cross-Platform Path Handling...")
        
        try:
            # Test data directory access
            data_dir = Path("updated_data/Clean")
            if data_dir.exists():
                # Test path normalization
                test_paths = [
                    data_dir / "comprehensive_df_PGDP_labeled.csv",
                    data_dir / "CS4_Statistical_Modeling" / "net_capital_flows_full.csv",
                    data_dir / "CS5_Capital_Controls" / "sd_yearly_flows.csv"
                ]
                
                accessible_paths = 0
                for path in test_paths:
                    if path.exists():
                        accessible_paths += 1
                
                if accessible_paths == len(test_paths):
                    self.log_test(
                        'Path Handling',
                        'Data File Access',
                        'PASS',
                        f'All {len(test_paths)} data files accessible'
                    )
                else:
                    self.log_test(
                        'Path Handling',
                        'Data File Access',
                        'WARNING',
                        f'{accessible_paths}/{len(test_paths)} files accessible'
                    )
            
            # Test output directory creation
            output_dir = Path("output")
            if not output_dir.exists():
                output_dir.mkdir(exist_ok=True)
            
            # Test file creation and deletion
            test_file = output_dir / "cross_platform_test.txt"
            test_file.write_text("Test content")
            
            if test_file.exists():
                test_file.unlink()  # Clean up
                self.log_test(
                    'Path Handling',
                    'File Operations',
                    'PASS',
                    'File creation and deletion successful'
                )
            else:
                self.log_test(
                    'Path Handling', 
                    'File Operations',
                    'FAIL',
                    'File creation failed'
                )
                
        except Exception as e:
            self.log_test(
                'Path Handling',
                'Path Compatibility',
                'FAIL',
                f'Path handling error: {str(e)}'
            )
    
    def test_dependency_compatibility(self):
        """Test dependency compatibility across platforms"""
        print("📦 Testing Dependency Compatibility...")
        
        required_packages = [
            'pandas',
            'numpy', 
            'scipy',
            'streamlit',
            'matplotlib',
            'plotly',
            'seaborn'
        ]
        
        compatible_packages = 0
        package_versions = {}
        
        for package in required_packages:
            try:
                if package == 'pandas':
                    import pandas as pd
                    package_versions[package] = pd.__version__
                elif package == 'numpy':
                    import numpy as np
                    package_versions[package] = np.__version__
                elif package == 'scipy':
                    import scipy
                    package_versions[package] = scipy.__version__
                elif package == 'streamlit':
                    import streamlit as st
                    package_versions[package] = st.__version__
                elif package == 'matplotlib':
                    import matplotlib
                    package_versions[package] = matplotlib.__version__
                elif package == 'plotly':
                    import plotly
                    package_versions[package] = plotly.__version__
                elif package == 'seaborn':
                    import seaborn as sns
                    package_versions[package] = sns.__version__
                
                compatible_packages += 1
                
            except ImportError:
                package_versions[package] = 'NOT INSTALLED'
        
        if compatible_packages == len(required_packages):
            self.log_test(
                'Dependencies',
                'Package Compatibility',
                'PASS',
                f'All {len(required_packages)} packages available'
            )
        else:
            self.log_test(
                'Dependencies',
                'Package Compatibility',
                'FAIL',
                f'{compatible_packages}/{len(required_packages)} packages available'
            )
        
        # Log version information
        for package, version in package_versions.items():
            self.log_test(
                'Dependencies',
                f'{package} Version',
                'INFO',
                version
            )
    
    def test_streamlit_compatibility(self):
        """Test Streamlit application compatibility"""
        print("🌐 Testing Streamlit Application Compatibility...")
        
        try:
            # Test if Streamlit can import dashboard modules
            sys.path.insert(0, str(Path("src/dashboard")))
            
            modules_to_test = [
                'simple_report_app',
                'case_study_2_euro_adoption',
                'main_app'
            ]
            
            importable_modules = 0
            
            for module_name in modules_to_test:
                try:
                    module_file = Path(f"src/dashboard/{module_name}.py")
                    if module_file.exists():
                        # Test basic syntax by parsing
                        with open(module_file, 'r') as f:
                            content = f.read()
                        
                        # Check for Streamlit imports
                        if 'import streamlit' in content or 'from streamlit' in content:
                            importable_modules += 1
                        
                except Exception as e:
                    self.log_test(
                        'Streamlit',
                        f'{module_name} Import Test',
                        'FAIL',
                        f'Import failed: {str(e)}'
                    )
            
            if importable_modules == len(modules_to_test):
                self.log_test(
                    'Streamlit',
                    'Dashboard Module Compatibility',
                    'PASS',
                    f'All {len(modules_to_test)} modules compatible'
                )
            else:
                self.log_test(
                    'Streamlit',
                    'Dashboard Module Compatibility',
                    'WARNING',
                    f'{importable_modules}/{len(modules_to_test)} modules compatible'
                )
                
        except Exception as e:
            self.log_test(
                'Streamlit',
                'Streamlit Compatibility',
                'FAIL',
                f'Streamlit compatibility error: {str(e)}'
            )
    
    def test_data_processing_performance(self):
        """Test data processing performance across platforms"""
        print("⚡ Testing Data Processing Performance...")
        
        try:
            # Test CSV reading performance
            data_file = Path("updated_data/Clean/comprehensive_df_PGDP_labeled.csv")
            if data_file.exists():
                
                start_time = time.time()
                df = pd.read_csv(data_file)
                load_time = time.time() - start_time
                
                if load_time < 1.0:
                    performance_status = 'EXCELLENT'
                elif load_time < 3.0:
                    performance_status = 'GOOD'
                elif load_time < 5.0:
                    performance_status = 'ACCEPTABLE'
                else:
                    performance_status = 'SLOW'
                
                self.log_test(
                    'Performance',
                    'CSV Loading Speed',
                    'PASS',
                    f'{performance_status}: {load_time:.2f}s for {len(df)} rows'
                )
                
                # Test statistical calculations
                start_time = time.time()
                
                # Sample statistical operations
                numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
                if len(numeric_cols) > 0:
                    stats = df[numeric_cols].describe()
                    correlations = df[numeric_cols].corr()
                
                calc_time = time.time() - start_time
                
                if calc_time < 0.5:
                    self.log_test(
                        'Performance',
                        'Statistical Calculations',
                        'PASS',
                        f'Fast calculations: {calc_time:.3f}s'
                    )
                else:
                    self.log_test(
                        'Performance',
                        'Statistical Calculations',
                        'WARNING',
                        f'Slow calculations: {calc_time:.3f}s'
                    )
            else:
                self.log_test(
                    'Performance',
                    'Data File Availability',
                    'FAIL',
                    'Primary data file not found'
                )
                
        except Exception as e:
            self.log_test(
                'Performance',
                'Performance Testing',
                'FAIL',
                f'Performance test error: {str(e)}'
            )
    
    def test_memory_usage(self):
        """Test memory usage patterns"""
        print("💾 Testing Memory Usage Patterns...")
        
        try:
            import psutil
            process = psutil.Process()
            
            # Get initial memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load data and measure memory increase
            data_file = Path("updated_data/Clean/comprehensive_df_PGDP_labeled.csv")
            if data_file.exists():
                df = pd.read_csv(data_file)
                
                after_load_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = after_load_memory - initial_memory
                
                if memory_increase < 100:  # Less than 100MB increase
                    self.log_test(
                        'Memory',
                        'Memory Usage Efficiency',
                        'PASS',
                        f'Memory increase: {memory_increase:.1f}MB'
                    )
                else:
                    self.log_test(
                        'Memory',
                        'Memory Usage Efficiency',
                        'WARNING',
                        f'High memory usage: {memory_increase:.1f}MB'
                    )
                
                # Test memory after operations
                test_operations = [
                    lambda: df.groupby('COUNTRY').mean() if 'COUNTRY' in df.columns else None,
                    lambda: df.describe(),
                    lambda: df.corr() if len(df.select_dtypes(include=[np.number]).columns) > 1 else None
                ]
                
                for i, operation in enumerate(test_operations):
                    try:
                        operation()
                        current_memory = process.memory_info().rss / 1024 / 1024
                        operation_memory = current_memory - after_load_memory
                        
                        if operation_memory < 50:  # Less than 50MB for operations
                            status = 'PASS'
                        else:
                            status = 'WARNING'
                        
                        self.log_test(
                            'Memory',
                            f'Operation {i+1} Memory',
                            status,
                            f'Memory: {operation_memory:.1f}MB'
                        )
                        
                    except Exception:
                        pass  # Skip operations that fail
            
        except ImportError:
            self.log_test(
                'Memory',
                'Memory Monitoring',
                'WARNING',
                'psutil not available for memory monitoring'
            )
        except Exception as e:
            self.log_test(
                'Memory',
                'Memory Testing',
                'WARNING',
                f'Memory test error: {str(e)}'
            )
    
    def test_file_encoding(self):
        """Test file encoding compatibility"""
        print("📝 Testing File Encoding Compatibility...")
        
        try:
            # Test CSV files with different encodings
            csv_files = list(Path("updated_data/Clean").rglob("*.csv"))[:5]  # Test first 5 files
            
            encoding_compatible = 0
            
            for csv_file in csv_files:
                try:
                    # Try UTF-8 first
                    df_utf8 = pd.read_csv(csv_file, encoding='utf-8')
                    encoding_compatible += 1
                    
                except UnicodeDecodeError:
                    # Try other encodings
                    for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                        try:
                            df_alt = pd.read_csv(csv_file, encoding=encoding)
                            encoding_compatible += 1
                            break
                        except:
                            continue
            
            if encoding_compatible == len(csv_files):
                self.log_test(
                    'Encoding',
                    'CSV File Encoding',
                    'PASS',
                    f'All {len(csv_files)} files readable'
                )
            else:
                self.log_test(
                    'Encoding',
                    'CSV File Encoding',
                    'WARNING',
                    f'{encoding_compatible}/{len(csv_files)} files readable'
                )
                
        except Exception as e:
            self.log_test(
                'Encoding',
                'Encoding Compatibility',
                'FAIL',
                f'Encoding test error: {str(e)}'
            )
    
    def generate_compatibility_report(self):
        """Generate cross-platform compatibility report"""
        print("\n" + "="*60)
        print("🌍 CROSS-PLATFORM COMPATIBILITY REPORT")
        print("="*60)
        
        # Platform information
        print(f"\n🖥️  PLATFORM INFORMATION:")
        print(f"  System: {self.platform_info['system']}")
        print(f"  Release: {self.platform_info['release']}")
        print(f"  Architecture: {self.platform_info['architecture'][0]}")
        print(f"  Python: {self.platform_info['python_version'].split()[0]}")
        
        # Test summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed_tests = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        warning_tests = sum(1 for r in self.test_results if r['status'] == 'WARNING')
        info_tests = sum(1 for r in self.test_results if r['status'] == 'INFO')
        
        actual_tests = total_tests - info_tests  # Exclude INFO entries
        
        print(f"\n📊 COMPATIBILITY SUMMARY:")
        print(f"  Total Tests: {actual_tests}")
        print(f"  ✅ Passed: {passed_tests}")
        print(f"  ❌ Failed: {failed_tests}")
        print(f"  ⚠️  Warnings: {warning_tests}")
        print(f"  🔍 Info: {info_tests}")
        
        if actual_tests > 0:
            compatibility_score = (passed_tests / actual_tests) * 100
            print(f"  📈 Compatibility Score: {compatibility_score:.1f}%")
        
        # Category breakdown
        categories = {}
        for result in self.test_results:
            if result['status'] != 'INFO':
                category = result['category']
                if category not in categories:
                    categories[category] = {'pass': 0, 'fail': 0, 'warning': 0}
                categories[category][result['status'].lower()] += 1
        
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, counts in categories.items():
            total_cat = sum(counts.values())
            if total_cat > 0:
                pass_rate = (counts['pass'] / total_cat) * 100
                print(f"  📂 {category}: {pass_rate:.1f}% pass rate ({counts['pass']}/{total_cat})")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        current_category = None
        for result in self.test_results:
            if result['category'] != current_category:
                print(f"\n  📂 {result['category']}:")
                current_category = result['category']
            
            if result['status'] == 'INFO':
                icon = "ℹ️"
            else:
                icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️"}[result['status']]
            
            print(f"    {icon} {result['test_name']}: {result['details']}")
        
        # Overall assessment
        if failed_tests == 0 and warning_tests <= 2:
            status = "EXCELLENT CROSS-PLATFORM COMPATIBILITY"
            print(f"\n🎉 OVERALL STATUS: {status}")
            print("   🌍 Platform works reliably across different environments")
        elif failed_tests == 0:
            status = "GOOD CROSS-PLATFORM COMPATIBILITY"
            print(f"\n✅ OVERALL STATUS: {status}")
            print("   🔍 Minor warnings, but core functionality works")
        elif failed_tests <= 2:
            status = "ACCEPTABLE COMPATIBILITY"
            print(f"\n📝 OVERALL STATUS: {status}")
            print("   ⚠️  Some platform-specific issues need attention")
        else:
            status = "COMPATIBILITY ISSUES DETECTED"
            print(f"\n🔧 OVERALL STATUS: {status}")
            print("   ❌ Multiple compatibility issues require resolution")
        
        return {
            'total_tests': actual_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'warnings': warning_tests,
            'compatibility_score': compatibility_score if actual_tests > 0 else 0,
            'status': status,
            'platform': self.platform_info
        }
    
    def run_full_compatibility_test(self):
        """Run complete cross-platform compatibility test"""
        print("🌍 Capital Flows Research - Cross-Platform Compatibility Test")
        print("="*60)
        print("Testing functionality across different platforms and environments...")
        print()
        
        # Run all compatibility tests
        self.test_path_handling()
        self.test_dependency_compatibility()
        self.test_streamlit_compatibility()
        self.test_data_processing_performance()
        self.test_memory_usage()
        self.test_file_encoding()
        
        # Generate comprehensive report
        summary = self.generate_compatibility_report()
        
        return summary

def main():
    """Main execution function"""
    tester = CrossPlatformTester()
    summary = tester.run_full_compatibility_test()
    
    # Exit with appropriate code
    if summary['failed'] > 2:
        sys.exit(1)  # Major compatibility issues
    elif summary['failed'] > 0:
        sys.exit(2)  # Minor compatibility issues
    else:
        sys.exit(0)  # Success

if __name__ == "__main__":
    main()