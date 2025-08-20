#!/usr/bin/env python3
"""
Final Production Readiness Report for Capital Flows Research Platform

Comprehensive assessment combining all audit results to provide 
definitive production readiness evaluation.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

class ProductionReadinessAssessment:
    """Final production readiness evaluation"""
    
    def __init__(self):
        self.audit_results = {}
        self.overall_scores = {}
        self.critical_issues = []
        self.recommendations = []
        
    def run_all_audits(self):
        """Execute all audit scripts and collect results"""
        print("🔍 Running Comprehensive Production Readiness Assessment")
        print("="*60)
        
        audit_scripts = [
            ("Testing Framework", "python run_tests.py"),
            ("Statistical Validation", "python statistical_validation.py"),
            ("UX/UI Quality", "python ux_ui_audit.py"),
            ("Style Consistency", "python style_consistency_enhancer.py"),
            ("Cross-Platform Compatibility", "python cross_platform_compatibility_test.py")
        ]
        
        for audit_name, command in audit_scripts:
            print(f"\n🧪 Running {audit_name}...")
            try:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )
                
                self.audit_results[audit_name] = {
                    'exit_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'success': result.returncode == 0
                }
                
                if result.returncode == 0:
                    print(f"  ✅ {audit_name}: PASSED")
                else:
                    print(f"  ❌ {audit_name}: FAILED (exit code: {result.returncode})")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏱️  {audit_name}: TIMEOUT")
                self.audit_results[audit_name] = {
                    'exit_code': -1,
                    'success': False,
                    'error': 'Timeout after 2 minutes'
                }
            except Exception as e:
                print(f"  ❌ {audit_name}: ERROR - {str(e)}")
                self.audit_results[audit_name] = {
                    'exit_code': -1,
                    'success': False,
                    'error': str(e)
                }
    
    def analyze_test_coverage(self):
        """Analyze testing framework coverage"""
        print("\n📊 Analyzing Test Coverage...")
        
        test_files = list(Path("tests").glob("test_*.py"))
        total_test_files = len(test_files)
        
        # Count test functions
        total_tests = 0
        for test_file in test_files:
            with open(test_file, 'r') as f:
                content = f.read()
            test_functions = len([line for line in content.split('\n') if line.strip().startswith('def test_')])
            total_tests += test_functions
        
        self.overall_scores['test_coverage'] = {
            'test_files': total_test_files,
            'total_tests': total_tests,
            'score': min(100, (total_tests / 50) * 100)  # 50 tests = 100% score
        }
        
        print(f"  📁 Test Files: {total_test_files}")
        print(f"  🧪 Total Tests: {total_tests}")
        print(f"  📈 Coverage Score: {self.overall_scores['test_coverage']['score']:.1f}%")
    
    def analyze_documentation_quality(self):
        """Analyze documentation completeness"""
        print("\n📚 Analyzing Documentation Quality...")
        
        doc_files = [
            ("CLAUDE.md", "Project Documentation"),
            ("README.md", "Main README"),
            ("tests/README.md", "Test Documentation"),
            ("STYLE_GUIDE.md", "Style Guide"),
            ("requirements.txt", "Dependencies")
        ]
        
        doc_score = 0
        doc_count = 0
        
        for file_path, description in doc_files:
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                if file_size > 1000:  # At least 1KB of content
                    doc_score += 20  # Each doc worth 20 points
                    print(f"  ✅ {description}: Complete")
                else:
                    doc_score += 10  # Partial credit for small files
                    print(f"  ⚠️  {description}: Minimal content")
                doc_count += 1
            else:
                print(f"  ❌ {description}: Missing")
        
        self.overall_scores['documentation'] = {
            'files_present': doc_count,
            'files_expected': len(doc_files),
            'score': doc_score
        }
        
        print(f"  📊 Documentation Score: {doc_score}%")
    
    def analyze_code_quality(self):
        """Analyze code quality metrics"""
        print("\n💻 Analyzing Code Quality...")
        
        python_files = list(Path("src").rglob("*.py"))
        total_lines = 0
        total_functions = 0
        total_classes = 0
        files_with_docstrings = 0
        
        for py_file in python_files:
            with open(py_file, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            total_lines += len(lines)
            
            # Count functions and classes
            total_functions += len([line for line in lines if line.strip().startswith('def ')])
            total_classes += len([line for line in lines if line.strip().startswith('class ')])
            
            # Check for docstrings
            if '"""' in content or "'''" in content:
                files_with_docstrings += 1
        
        code_quality_score = 0
        
        # Lines of code (50 points max)
        if total_lines >= 1000:
            code_quality_score += 50
        else:
            code_quality_score += (total_lines / 1000) * 50
        
        # Documentation (30 points max)
        doc_percentage = (files_with_docstrings / len(python_files)) * 100 if python_files else 0
        code_quality_score += (doc_percentage / 100) * 30
        
        # Structure (20 points max)
        if total_classes >= 5 and total_functions >= 20:
            code_quality_score += 20
        else:
            code_quality_score += ((total_classes + total_functions) / 25) * 20
        
        self.overall_scores['code_quality'] = {
            'total_lines': total_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'documentation_percentage': doc_percentage,
            'score': min(100, code_quality_score)
        }
        
        print(f"  📄 Python Files: {len(python_files)}")
        print(f"  📏 Lines of Code: {total_lines}")
        print(f"  🔧 Functions: {total_functions}")
        print(f"  🏗️  Classes: {total_classes}")
        print(f"  📖 Documentation: {doc_percentage:.1f}%")
        print(f"  📊 Code Quality Score: {code_quality_score:.1f}%")
    
    def assess_production_readiness(self):
        """Generate final production readiness assessment"""
        print("\n" + "="*60)
        print("🎯 FINAL PRODUCTION READINESS ASSESSMENT")
        print("="*60)
        
        # Calculate overall score
        weights = {
            'test_coverage': 0.25,
            'documentation': 0.20,
            'code_quality': 0.20,
            'audit_success': 0.35
        }
        
        # Audit success score
        successful_audits = sum(1 for result in self.audit_results.values() if result['success'])
        total_audits = len(self.audit_results)
        audit_success_score = (successful_audits / total_audits) * 100 if total_audits > 0 else 0
        
        # Calculate weighted overall score
        overall_score = (
            self.overall_scores['test_coverage']['score'] * weights['test_coverage'] +
            self.overall_scores['documentation']['score'] * weights['documentation'] +
            self.overall_scores['code_quality']['score'] * weights['code_quality'] +
            audit_success_score * weights['audit_success']
        )
        
        print(f"\n📊 READINESS METRICS:")
        print(f"  🧪 Test Coverage: {self.overall_scores['test_coverage']['score']:.1f}%")
        print(f"  📚 Documentation: {self.overall_scores['documentation']['score']:.1f}%")
        print(f"  💻 Code Quality: {self.overall_scores['code_quality']['score']:.1f}%")
        print(f"  ✅ Audit Success: {audit_success_score:.1f}%")
        print(f"  🎯 OVERALL SCORE: {overall_score:.1f}%")
        
        # Audit results summary
        print(f"\n🔍 AUDIT RESULTS SUMMARY:")
        for audit_name, result in self.audit_results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"  {status} {audit_name}")
        
        # Production readiness classification
        if overall_score >= 90:
            readiness_level = "🚀 PRODUCTION READY"
            readiness_description = "Excellent quality - Ready for immediate deployment"
        elif overall_score >= 80:
            readiness_level = "✅ NEAR PRODUCTION READY"
            readiness_description = "High quality - Minor improvements recommended"
        elif overall_score >= 70:
            readiness_level = "📝 NEEDS IMPROVEMENTS"
            readiness_description = "Good foundation - Address key issues before production"
        elif overall_score >= 60:
            readiness_level = "⚠️  SIGNIFICANT IMPROVEMENTS NEEDED"
            readiness_description = "Major issues - Extensive work required"
        else:
            readiness_level = "❌ NOT PRODUCTION READY"
            readiness_description = "Critical issues - Complete overhaul needed"
        
        print(f"\n{readiness_level}")
        print(f"📋 Assessment: {readiness_description}")
        
        # Specific recommendations
        print(f"\n💡 KEY RECOMMENDATIONS:")
        
        if self.overall_scores['test_coverage']['score'] < 80:
            print("  🧪 Expand test coverage - Add more comprehensive test cases")
        
        if self.overall_scores['documentation']['score'] < 80:
            print("  📚 Improve documentation - Add missing documentation files")
        
        if self.overall_scores['code_quality']['score'] < 80:
            print("  💻 Enhance code quality - Add docstrings and improve structure")
        
        if audit_success_score < 100:
            failed_audits = [name for name, result in self.audit_results.items() if not result['success']]
            print(f"  🔧 Fix audit failures: {', '.join(failed_audits)}")
        
        if overall_score >= 90:
            print("  🎉 Platform exceeds production standards!")
            print("  🚀 Ready for academic research and policy applications")
        
        # Generate deployment checklist
        print(f"\n📋 DEPLOYMENT CHECKLIST:")
        checklist_items = [
            ("✅" if self.overall_scores['test_coverage']['score'] >= 80 else "❌", "Comprehensive test suite"),
            ("✅" if self.overall_scores['documentation']['score'] >= 80 else "❌", "Complete documentation"),
            ("✅" if successful_audits == total_audits else "❌", "All audits passing"),
            ("✅" if Path("requirements.txt").exists() else "❌", "Dependencies documented"),
            ("✅" if Path("CLAUDE.md").exists() else "❌", "Project instructions available"),
            ("✅" if list(Path("src/dashboard").glob("*.py")) else "❌", "Dashboard applications ready")
        ]
        
        for status, item in checklist_items:
            print(f"  {status} {item}")
        
        return {
            'overall_score': overall_score,
            'readiness_level': readiness_level,
            'audit_results': self.audit_results,
            'scores': self.overall_scores,
            'recommendations': self.recommendations
        }
    
    def run_complete_assessment(self):
        """Run complete production readiness assessment"""
        print("🎯 Capital Flows Research - Final Production Readiness Assessment")
        print("="*80)
        print("Comprehensive evaluation of platform readiness for academic research use")
        print()
        
        # Run all components
        self.run_all_audits()
        self.analyze_test_coverage()
        self.analyze_documentation_quality()
        self.analyze_code_quality()
        
        # Generate final assessment
        final_assessment = self.assess_production_readiness()
        
        return final_assessment

def main():
    """Main execution function"""
    assessor = ProductionReadinessAssessment()
    assessment = assessor.run_complete_assessment()
    
    # Exit with code based on readiness
    if assessment['overall_score'] >= 90:
        sys.exit(0)  # Production ready
    elif assessment['overall_score'] >= 80:
        sys.exit(1)  # Near production ready
    else:
        sys.exit(2)  # Needs improvements

if __name__ == "__main__":
    main()