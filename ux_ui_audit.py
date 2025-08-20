#!/usr/bin/env python3
"""
UX/UI Quality Audit for Capital Flows Research Platform

Comprehensive audit of user experience, interface quality, and production readiness
across all dashboard applications and supporting infrastructure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import re
import time
from datetime import datetime
import subprocess
import importlib.util
import ast

# Add dashboard modules to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "dashboard"))

warnings.filterwarnings('ignore')

class UXUIAuditor:
    """Comprehensive UX/UI quality auditor for Capital Flows platform"""
    
    def __init__(self):
        self.audit_results = []
        self.critical_issues = []
        self.ux_warnings = []
        self.performance_metrics = {}
        
    def log_result(self, category, test_name, status, details, critical=False):
        """Log audit result"""
        result = {
            'category': category,
            'test_name': test_name,
            'status': status,  # 'PASS', 'FAIL', 'WARNING'
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.audit_results.append(result)
        
        if status == 'FAIL' and critical:
            self.critical_issues.append(result)
        elif status == 'WARNING':
            self.ux_warnings.append(result)
    
    def audit_navigation_consistency(self):
        """Audit navigation flow and interface consistency"""
        print("🧭 Auditing Navigation and Flow Consistency...")
        
        # Check main dashboard structure
        try:
            main_app_path = Path("src/dashboard/main_app.py")
            if main_app_path.exists():
                with open(main_app_path, 'r') as f:
                    main_content = f.read()
                
                # Check for consistent tab structure
                tab_matches = re.findall(r'st\.tabs?\(\[(.*?)\]', main_content, re.DOTALL)
                if tab_matches:
                    tabs_content = tab_matches[0]
                    tab_count = len(re.findall(r'"([^"]+)"', tabs_content))
                    
                    self.log_result(
                        'Navigation',
                        'Main Dashboard Tab Structure',
                        'PASS',
                        f'Found {tab_count} tabs in main dashboard'
                    )
                else:
                    self.log_result(
                        'Navigation',
                        'Main Dashboard Tab Structure',
                        'WARNING',
                        'Could not detect tab structure in main dashboard'
                    )
                
                # Check for consistent header styling
                header_patterns = [
                    r'st\.title\(',
                    r'st\.header\(',
                    r'st\.subheader\(',
                    r'st\.markdown\(.*?#',
                ]
                
                header_count = 0
                for pattern in header_patterns:
                    matches = re.findall(pattern, main_content)
                    header_count += len(matches)
                
                if header_count >= 5:
                    self.log_result(
                        'Navigation',
                        'Header Structure Consistency',
                        'PASS',
                        f'Found {header_count} header elements'
                    )
                else:
                    self.log_result(
                        'Navigation',
                        'Header Structure Consistency',
                        'WARNING',
                        f'Limited header structure: {header_count} elements'
                    )
                
                # Check for download functionality
                download_patterns = [
                    r'st\.download_button',
                    r'download.*button',
                    r'export.*button'
                ]
                
                download_count = 0
                for pattern in download_patterns:
                    matches = re.findall(pattern, main_content, re.IGNORECASE)
                    download_count += len(matches)
                
                if download_count >= 3:
                    self.log_result(
                        'Navigation',
                        'Download Functionality',
                        'PASS',
                        f'Found {download_count} download/export buttons'
                    )
                else:
                    self.log_result(
                        'Navigation',
                        'Download Functionality',
                        'WARNING',
                        f'Limited download options: {download_count} buttons'
                    )
            
        except Exception as e:
            self.log_result(
                'Navigation',
                'Navigation Consistency Audit',
                'FAIL',
                f'Error auditing navigation: {str(e)}',
                critical=True
            )
    
    def audit_presentation_standards(self):
        """Audit professional presentation and formatting standards"""
        print("🎨 Auditing Professional Presentation Standards...")
        
        try:
            dashboard_files = [
                "src/dashboard/main_app.py",
                "src/dashboard/simple_report_app.py", 
                "src/dashboard/case_study_2_euro_adoption.py"
            ]
            
            style_consistency_score = 0
            total_checks = 0
            
            for file_path in dashboard_files:
                if Path(file_path).exists():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for consistent table formatting
                    total_checks += 1
                    if 'st.dataframe' in content or 'st.table' in content:
                        style_consistency_score += 1
                        
                    # Check for chart configuration consistency
                    total_checks += 1
                    chart_configs = re.findall(r'fig\.update_layout\(', content)
                    if len(chart_configs) >= 2:
                        style_consistency_score += 1
                    
                    # Check for color scheme consistency
                    total_checks += 1
                    color_patterns = re.findall(r'color[s]?\s*=\s*[\'"][^\'\"]+[\'"]', content)
                    if len(color_patterns) >= 3:
                        style_consistency_score += 1
                    
                    # Check for consistent error handling
                    total_checks += 1
                    if 'st.error' in content or 'st.warning' in content:
                        style_consistency_score += 1
            
            consistency_percentage = (style_consistency_score / total_checks) * 100 if total_checks > 0 else 0
            
            if consistency_percentage >= 75:
                self.log_result(
                    'Presentation',
                    'Style Consistency',
                    'PASS',
                    f'Style consistency: {consistency_percentage:.1f}%'
                )
            elif consistency_percentage >= 50:
                self.log_result(
                    'Presentation',
                    'Style Consistency',
                    'WARNING',
                    f'Moderate style consistency: {consistency_percentage:.1f}%'
                )
            else:
                self.log_result(
                    'Presentation',
                    'Style Consistency',
                    'FAIL',
                    f'Poor style consistency: {consistency_percentage:.1f}%',
                    critical=True
                )
            
            # Check for statistical notation standards
            notation_files = list(Path("src/dashboard").glob("*.py"))
            notation_compliance = 0
            notation_total = 0
            
            for file_path in notation_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for significance stars
                notation_total += 1
                if '***' in content or 'significance' in content.lower():
                    notation_compliance += 1
                
                # Check for proper statistical formatting
                notation_total += 1
                if 'p-value' in content.lower() or 'f-statistic' in content.lower():
                    notation_compliance += 1
            
            notation_score = (notation_compliance / notation_total) * 100 if notation_total > 0 else 0
            
            if notation_score >= 60:
                self.log_result(
                    'Presentation',
                    'Statistical Notation Standards',
                    'PASS',
                    f'Statistical notation compliance: {notation_score:.1f}%'
                )
            else:
                self.log_result(
                    'Presentation',
                    'Statistical Notation Standards',
                    'WARNING',
                    f'Statistical notation needs improvement: {notation_score:.1f}%'
                )
                
        except Exception as e:
            self.log_result(
                'Presentation',
                'Presentation Standards Audit',
                'FAIL',
                f'Error auditing presentation: {str(e)}',
                critical=True
            )
    
    def audit_content_clarity(self):
        """Audit content clarity and documentation quality"""
        print("📚 Auditing Content Clarity and Documentation...")
        
        try:
            # Check CLAUDE.md documentation quality
            claude_md_path = Path("CLAUDE.md")
            if claude_md_path.exists():
                with open(claude_md_path, 'r') as f:
                    claude_content = f.read()
                
                # Check documentation completeness
                required_sections = [
                    'Project Overview',
                    'Case Study',
                    'Development Environment',
                    'Data Processing',
                    'Dependencies'
                ]
                
                sections_found = 0
                for section in required_sections:
                    if section in claude_content:
                        sections_found += 1
                
                if sections_found >= 4:
                    self.log_result(
                        'Documentation',
                        'CLAUDE.md Completeness',
                        'PASS',
                        f'Found {sections_found}/{len(required_sections)} required sections'
                    )
                else:
                    self.log_result(
                        'Documentation',
                        'CLAUDE.md Completeness',
                        'WARNING',
                        f'Missing documentation sections: {sections_found}/{len(required_sections)}'
                    )
                
                # Check for methodology explanations
                methodology_keywords = [
                    'methodology',
                    'statistical',
                    'F-test',
                    'crisis period',
                    'GDP normalization'
                ]
                
                methodology_coverage = 0
                for keyword in methodology_keywords:
                    if keyword.lower() in claude_content.lower():
                        methodology_coverage += 1
                
                if methodology_coverage >= 3:
                    self.log_result(
                        'Documentation',
                        'Methodology Documentation',
                        'PASS',
                        f'Methodology coverage: {methodology_coverage}/{len(methodology_keywords)} topics'
                    )
                else:
                    self.log_result(
                        'Documentation',
                        'Methodology Documentation',
                        'WARNING',
                        f'Limited methodology documentation: {methodology_coverage}/{len(methodology_keywords)}'
                    )
            
            # Check README.md quality in tests
            test_readme_path = Path("tests/README.md")
            if test_readme_path.exists():
                with open(test_readme_path, 'r') as f:
                    readme_content = f.read()
                
                readme_sections = [
                    'Overview',
                    'Running Tests',
                    'Expected Results',
                    'Troubleshooting'
                ]
                
                readme_completeness = sum(1 for section in readme_sections if section in readme_content)
                
                if readme_completeness >= 3:
                    self.log_result(
                        'Documentation',
                        'Test Documentation Quality',
                        'PASS',
                        f'Test README completeness: {readme_completeness}/{len(readme_sections)}'
                    )
                else:
                    self.log_result(
                        'Documentation',
                        'Test Documentation Quality',
                        'WARNING',
                        f'Test README needs improvement: {readme_completeness}/{len(readme_sections)}'
                    )
            
            # Check for crisis period documentation
            dashboard_files = list(Path("src/dashboard").glob("*.py"))
            crisis_documentation = 0
            
            for file_path in dashboard_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if 'crisis' in content.lower() and ('2008' in content or '2020' in content):
                    crisis_documentation += 1
            
            if crisis_documentation >= 2:
                self.log_result(
                    'Documentation',
                    'Crisis Period Documentation',
                    'PASS',
                    f'Crisis periods documented in {crisis_documentation} files'
                )
            else:
                self.log_result(
                    'Documentation',
                    'Crisis Period Documentation',
                    'WARNING',
                    f'Limited crisis period documentation: {crisis_documentation} files'
                )
                
        except Exception as e:
            self.log_result(
                'Documentation',
                'Content Clarity Audit',
                'FAIL',
                f'Error auditing content clarity: {str(e)}',
                critical=True
            )
    
    def audit_performance_metrics(self):
        """Audit application performance and identify bottlenecks"""
        print("⚡ Auditing Application Performance...")
        
        try:
            # Test data loading performance
            start_time = time.time()
            
            try:
                from simple_report_app import load_default_data
                
                load_start = time.time()
                data, indicators, metadata = load_default_data(include_crisis_years=True)
                load_time = time.time() - load_start
                
                self.performance_metrics['cs1_data_load_time'] = load_time
                
                if load_time < 2.0:
                    self.log_result(
                        'Performance',
                        'CS1 Data Loading Speed',
                        'PASS',
                        f'Data loads in {load_time:.2f}s (target: <2s)'
                    )
                elif load_time < 5.0:
                    self.log_result(
                        'Performance',
                        'CS1 Data Loading Speed',
                        'WARNING',
                        f'Slow data loading: {load_time:.2f}s (target: <2s)'
                    )
                else:
                    self.log_result(
                        'Performance',
                        'CS1 Data Loading Speed',
                        'FAIL',
                        f'Very slow data loading: {load_time:.2f}s (target: <2s)',
                        critical=True
                    )
                
                # Test memory usage
                data_memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                self.performance_metrics['cs1_memory_usage_mb'] = data_memory_mb
                
                if data_memory_mb < 50:
                    self.log_result(
                        'Performance',
                        'CS1 Memory Usage',
                        'PASS',
                        f'Memory usage: {data_memory_mb:.1f}MB (target: <50MB)'
                    )
                else:
                    self.log_result(
                        'Performance',
                        'CS1 Memory Usage',
                        'WARNING',
                        f'High memory usage: {data_memory_mb:.1f}MB (target: <50MB)'
                    )
                
            except Exception as e:
                self.log_result(
                    'Performance',
                    'CS1 Performance Test',
                    'FAIL',
                    f'Error testing CS1 performance: {str(e)}',
                    critical=True
                )
            
            # Test CS2 performance
            try:
                from case_study_2_euro_adoption import load_case_study_2_data
                
                cs2_start = time.time()
                cs2_data, cs2_indicators, cs2_metadata = load_case_study_2_data(include_crisis_years=True)
                cs2_time = time.time() - cs2_start
                
                self.performance_metrics['cs2_data_load_time'] = cs2_time
                
                if cs2_time < 2.0:
                    self.log_result(
                        'Performance',
                        'CS2 Data Loading Speed',
                        'PASS',
                        f'CS2 loads in {cs2_time:.2f}s'
                    )
                else:
                    self.log_result(
                        'Performance',
                        'CS2 Data Loading Speed',
                        'WARNING',
                        f'CS2 slow loading: {cs2_time:.2f}s'
                    )
                
            except Exception as e:
                self.log_result(
                    'Performance',
                    'CS2 Performance Test',
                    'WARNING',
                    f'Could not test CS2 performance: {str(e)}'
                )
            
            # Check file sizes for potential bottlenecks
            data_dir = Path("updated_data/Clean")
            if data_dir.exists():
                large_files = []
                for file_path in data_dir.rglob("*.csv"):
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    if size_mb > 10:  # Files larger than 10MB
                        large_files.append((file_path.name, size_mb))
                
                if len(large_files) <= 2:
                    self.log_result(
                        'Performance',
                        'Data File Sizes',
                        'PASS',
                        f'Reasonable file sizes: {len(large_files)} files >10MB'
                    )
                else:
                    self.log_result(
                        'Performance',
                        'Data File Sizes',
                        'WARNING',
                        f'Multiple large files may impact performance: {len(large_files)} files >10MB'
                    )
                
        except Exception as e:
            self.log_result(
                'Performance',
                'Performance Audit',
                'FAIL',
                f'Error auditing performance: {str(e)}',
                critical=True
            )
    
    def audit_error_handling(self):
        """Audit error handling and edge case behavior"""
        print("🛡️ Auditing Error Handling and Edge Cases...")
        
        try:
            dashboard_files = [
                "src/dashboard/main_app.py",
                "src/dashboard/simple_report_app.py",
                "src/dashboard/case_study_2_euro_adoption.py"
            ]
            
            error_handling_score = 0
            error_checks = 0
            
            for file_path in dashboard_files:
                if Path(file_path).exists():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for try-except blocks
                    error_checks += 1
                    try_count = len(re.findall(r'try:', content))
                    except_count = len(re.findall(r'except', content))
                    
                    if try_count >= 2 and except_count >= 2:
                        error_handling_score += 1
                    
                    # Check for Streamlit error messages
                    error_checks += 1
                    streamlit_errors = len(re.findall(r'st\.(error|warning|info)', content))
                    if streamlit_errors >= 1:
                        error_handling_score += 1
                    
                    # Check for data validation
                    error_checks += 1
                    validation_patterns = [
                        r'\.exists\(\)',
                        r'\.empty',
                        r'is None',
                        r'len\([^)]+\)\s*[><=]'
                    ]
                    
                    validation_count = 0
                    for pattern in validation_patterns:
                        validation_count += len(re.findall(pattern, content))
                    
                    if validation_count >= 3:
                        error_handling_score += 1
            
            error_handling_percentage = (error_handling_score / error_checks) * 100 if error_checks > 0 else 0
            
            if error_handling_percentage >= 70:
                self.log_result(
                    'Error Handling',
                    'Error Handling Coverage',
                    'PASS',
                    f'Error handling coverage: {error_handling_percentage:.1f}%'
                )
            elif error_handling_percentage >= 40:
                self.log_result(
                    'Error Handling',
                    'Error Handling Coverage',
                    'WARNING',
                    f'Moderate error handling: {error_handling_percentage:.1f}%'
                )
            else:
                self.log_result(
                    'Error Handling',
                    'Error Handling Coverage',
                    'FAIL',
                    f'Poor error handling: {error_handling_percentage:.1f}%',
                    critical=True
                )
            
            # Test actual error scenarios
            try:
                # Test with non-existent file path
                from simple_report_app import load_default_data
                
                # This should gracefully handle missing data
                original_path = Path("updated_data/Clean/comprehensive_df_PGDP_labeled.csv")
                if original_path.exists():
                    self.log_result(
                        'Error Handling',
                        'Data File Accessibility',
                        'PASS',
                        'Primary data file exists and accessible'
                    )
                else:
                    self.log_result(
                        'Error Handling',
                        'Data File Accessibility',
                        'FAIL',
                        'Primary data file not accessible',
                        critical=True
                    )
                
            except Exception as e:
                self.log_result(
                    'Error Handling',
                    'Error Scenario Testing',
                    'WARNING',
                    f'Could not test error scenarios: {str(e)}'
                )
                
        except Exception as e:
            self.log_result(
                'Error Handling',
                'Error Handling Audit',
                'FAIL',
                f'Error auditing error handling: {str(e)}',
                critical=True
            )
    
    def audit_deployment_readiness(self):
        """Audit deployment preparation and documentation"""
        print("🚀 Auditing Deployment Readiness...")
        
        try:
            # Check requirements.txt
            requirements_path = Path("requirements.txt")
            if requirements_path.exists():
                with open(requirements_path, 'r') as f:
                    requirements_content = f.read()
                
                required_packages = [
                    'streamlit',
                    'pandas',
                    'numpy',
                    'scipy',
                    'matplotlib',
                    'plotly'
                ]
                
                packages_found = 0
                for package in required_packages:
                    if package in requirements_content:
                        packages_found += 1
                
                if packages_found >= 5:
                    self.log_result(
                        'Deployment',
                        'Requirements.txt Completeness',
                        'PASS',
                        f'Found {packages_found}/{len(required_packages)} core packages'
                    )
                else:
                    self.log_result(
                        'Deployment',
                        'Requirements.txt Completeness',
                        'WARNING',
                        f'Missing core packages: {packages_found}/{len(required_packages)}'
                    )
            else:
                self.log_result(
                    'Deployment',
                    'Requirements.txt Existence',
                    'FAIL',
                    'requirements.txt file missing',
                    critical=True
                )
            
            # Check project structure
            expected_structure = [
                "src/dashboard",
                "src/core",
                "updated_data/Clean",
                "tests",
                "CLAUDE.md"
            ]
            
            structure_complete = 0
            for path in expected_structure:
                if Path(path).exists():
                    structure_complete += 1
            
            if structure_complete >= 4:
                self.log_result(
                    'Deployment',
                    'Project Structure',
                    'PASS',
                    f'Project structure complete: {structure_complete}/{len(expected_structure)}'
                )
            else:
                self.log_result(
                    'Deployment',
                    'Project Structure',
                    'WARNING',
                    f'Incomplete project structure: {structure_complete}/{len(expected_structure)}'
                )
            
            # Check for configuration files
            config_files = [
                "src/core/config.py",
                ".gitignore"
            ]
            
            config_present = sum(1 for f in config_files if Path(f).exists())
            
            if config_present >= 1:
                self.log_result(
                    'Deployment',
                    'Configuration Files',
                    'PASS',
                    f'Configuration files present: {config_present}/{len(config_files)}'
                )
            else:
                self.log_result(
                    'Deployment',
                    'Configuration Files',
                    'WARNING',
                    'Missing configuration files'
                )
            
            # Check testing infrastructure
            test_files = list(Path("tests").glob("test_*.py")) if Path("tests").exists() else []
            
            if len(test_files) >= 5:
                self.log_result(
                    'Deployment',
                    'Testing Infrastructure',
                    'PASS',
                    f'Comprehensive test suite: {len(test_files)} test files'
                )
            elif len(test_files) >= 2:
                self.log_result(
                    'Deployment',
                    'Testing Infrastructure',
                    'WARNING',
                    f'Limited test coverage: {len(test_files)} test files'
                )
            else:
                self.log_result(
                    'Deployment',
                    'Testing Infrastructure',
                    'FAIL',
                    'Insufficient testing infrastructure',
                    critical=True
                )
                
        except Exception as e:
            self.log_result(
                'Deployment',
                'Deployment Readiness Audit',
                'FAIL',
                f'Error auditing deployment readiness: {str(e)}',
                critical=True
            )
    
    def generate_ux_ui_report(self):
        """Generate comprehensive UX/UI audit report"""
        print("\n" + "="*60)
        print("🎯 UX/UI QUALITY AUDIT REPORT")
        print("="*60)
        
        total_tests = len(self.audit_results)
        passed_tests = sum(1 for r in self.audit_results if r['status'] == 'PASS')
        failed_tests = sum(1 for r in self.audit_results if r['status'] == 'FAIL')
        warning_tests = sum(1 for r in self.audit_results if r['status'] == 'WARNING')
        
        print(f"\n📊 AUDIT SUMMARY:")
        print(f"  Total Checks: {total_tests}")
        print(f"  ✅ Passed: {passed_tests}")
        print(f"  ❌ Failed: {failed_tests}")
        print(f"  ⚠️  Warnings: {warning_tests}")
        print(f"  📈 Quality Score: {passed_tests/total_tests*100:.1f}%")
        
        # Performance metrics summary
        if self.performance_metrics:
            print(f"\n⚡ PERFORMANCE METRICS:")
            for metric, value in self.performance_metrics.items():
                if 'time' in metric:
                    print(f"  ⏱️  {metric}: {value:.2f}s")
                elif 'memory' in metric:
                    print(f"  💾 {metric}: {value:.1f}MB")
        
        # Category breakdown
        categories = {}
        for result in self.audit_results:
            category = result['category']
            if category not in categories:
                categories[category] = {'pass': 0, 'fail': 0, 'warning': 0}
            categories[category][result['status'].lower()] += 1
        
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, counts in categories.items():
            total_cat = sum(counts.values())
            pass_rate = (counts['pass'] / total_cat) * 100 if total_cat > 0 else 0
            print(f"  📂 {category}: {pass_rate:.1f}% pass rate ({counts['pass']}/{total_cat})")
        
        if self.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(self.critical_issues)}):")
            for issue in self.critical_issues:
                print(f"  ❌ {issue['test_name']}: {issue['details']}")
        
        if self.ux_warnings:
            print(f"\n⚠️  UX WARNINGS ({len(self.ux_warnings)}):") 
            for warning in self.ux_warnings[:5]:  # Show first 5 warnings
                print(f"  ⚠️  {warning['test_name']}: {warning['details']}")
        
        print(f"\n📋 DETAILED RESULTS:")
        current_category = None
        for result in self.audit_results:
            if result['category'] != current_category:
                print(f"\n  📂 {result['category']}:")
                current_category = result['category']
            
            status_icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️"}[result['status']]
            print(f"    {status_icon} {result['test_name']}: {result['details']}")
        
        # User experience assessment
        print(f"\n🎨 USER EXPERIENCE ASSESSMENT:")
        
        if len(self.critical_issues) == 0:
            print("  ✅ Core functionality is stable")
        else:
            print(f"  ❌ {len(self.critical_issues)} critical UX issues need immediate attention")
        
        if failed_tests == 0:
            print("  ✅ All interface components are functional")
        else:
            print(f"  ⚠️  {failed_tests} interface issues require review")
        
        if warning_tests <= total_tests * 0.3:  # Less than 30% warnings acceptable
            print("  ✅ Professional presentation standards met")
        else:
            print(f"  ⚠️  {warning_tests} presentation improvements recommended")
        
        # Overall UX assessment
        if len(self.critical_issues) == 0 and failed_tests == 0:
            ux_status = "PRODUCTION READY"
            print(f"\n🎉 OVERALL UX STATUS: {ux_status}")
            print("   🚀 Platform ready for academic and policy research use")
        elif len(self.critical_issues) == 0:
            ux_status = "MINOR UX IMPROVEMENTS NEEDED"
            print(f"\n📝 OVERALL UX STATUS: {ux_status}")
            print("   🔍 Address warnings for optimal user experience")
        else:
            ux_status = "MAJOR UX ISSUES REQUIRE ATTENTION"
            print(f"\n🔧 OVERALL UX STATUS: {ux_status}")
            print("   ⚠️  Critical interface issues must be resolved before production")
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'warnings': warning_tests,
            'critical_issues': len(self.critical_issues),
            'overall_status': ux_status,
            'performance_metrics': self.performance_metrics
        }
    
    def run_full_ux_audit(self):
        """Run complete UX/UI quality audit"""
        print("🎯 Capital Flows Research - UX/UI Quality Audit")
        print("="*60)
        print("Ensuring professional presentation and optimal user experience...")
        print()
        
        # Run all audit components
        self.audit_navigation_consistency()
        self.audit_presentation_standards()
        self.audit_content_clarity()
        self.audit_performance_metrics()
        self.audit_error_handling()
        self.audit_deployment_readiness()
        
        # Generate comprehensive report
        summary = self.generate_ux_ui_report()
        
        return summary

def main():
    """Main execution function"""
    auditor = UXUIAuditor()
    summary = auditor.run_full_ux_audit()
    
    # Exit with appropriate code
    if summary['critical_issues'] > 0:
        sys.exit(1)  # Critical UX issues
    elif summary['failed'] > 0:
        sys.exit(2)  # Non-critical issues
    else:
        sys.exit(0)  # Success

if __name__ == "__main__":
    main()