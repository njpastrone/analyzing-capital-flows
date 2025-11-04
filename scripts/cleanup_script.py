#!/usr/bin/env python3
"""
Codebase Cleanup Script - Remove deprecated files and optimize project structure
Run this script to clean up unused files and standardize the project structure.

IMPORTANT: Review the files to be deleted before running with --execute flag.
"""

import os
import shutil
from pathlib import Path
import argparse

# Files and directories to potentially remove (review before executing)
DEPRECATED_FILES = [
    # Unused core modules (replaced by centralized functions)
    "src/core/data_processor.py",
    "src/core/statistical_tests.py", 
    "src/core/visualizer.py",
    "src/core/config.py",
    
    # Template files (not used in production)
    "src/dashboard/templates/case_study_template.py",
    
    # Legacy data files (replaced by updated_data structure)
    "data/case_one_grouped.csv",
    "data/case_study_2_gdp_data.csv",
    
    # Development/testing files
    "test_main_app.py",
]

POTENTIALLY_EMPTY_DIRS = [
    "src/dashboard/templates",
    "output"  # May be empty if no reports generated yet
]

LEGACY_NOTEBOOKS = [
    # Case study one legacy files (analysis moved to Streamlit)
    "src/case_study_one/Case_Study_1_Analysis.ipynb",
    "src/case_study_one/Case_Study_1_Debug_Analysis.ipynb",
    "src/case_study_one/Case_Study_1_Report_Template.ipynb",
    "src/case_study_one/cleaning_case_one.py",
]

def check_file_usage(file_path: str, project_root: Path) -> bool:
    """Check if a file is still being imported/used in the project"""
    file_name = Path(file_path).stem
    
    # Search for imports of this file
    for py_file in project_root.rglob("*.py"):
        if py_file.name == Path(file_path).name:
            continue  # Skip the file itself
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if f"import {file_name}" in content or f"from {file_name}" in content:
                    return True
        except:
            continue
    
    return False

def get_file_size(file_path: Path) -> int:
    """Get file size in bytes"""
    try:
        return file_path.stat().st_size
    except:
        return 0

def main():
    parser = argparse.ArgumentParser(description='Cleanup Capital Flows Research codebase')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually delete files (default: dry run)')
    parser.add_argument('--include-notebooks', action='store_true',
                       help='Include legacy Jupyter notebooks in cleanup')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    total_size_saved = 0
    files_to_remove = []
    
    print("🔍 Capital Flows Research - Codebase Cleanup Analysis")
    print("=" * 60)
    
    # Check deprecated files
    print("\n📁 DEPRECATED FILES ANALYSIS:")
    for file_path in DEPRECATED_FILES:
        full_path = project_root / file_path
        if full_path.exists():
            is_used = check_file_usage(file_path, project_root)
            file_size = get_file_size(full_path)
            
            status = "🔗 STILL USED" if is_used else "❌ UNUSED"
            print(f"{status:15} {file_path:50} ({file_size:,} bytes)")
            
            if not is_used:
                files_to_remove.append(full_path)
                total_size_saved += file_size
        else:
            print(f"{'✅ NOT FOUND':15} {file_path:50}")
    
    # Check legacy notebooks if requested
    if args.include_notebooks:
        print("\n📓 LEGACY NOTEBOOKS ANALYSIS:")
        for file_path in LEGACY_NOTEBOOKS:
            full_path = project_root / file_path
            if full_path.exists():
                file_size = get_file_size(full_path)
                print(f"{'📝 NOTEBOOK':15} {file_path:50} ({file_size:,} bytes)")
                files_to_remove.append(full_path)
                total_size_saved += file_size
    
    # Check empty directories
    print("\n📂 EMPTY DIRECTORIES ANALYSIS:")
    for dir_path in POTENTIALLY_EMPTY_DIRS:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            try:
                files_in_dir = list(full_path.rglob("*"))
                if len(files_in_dir) == 0:
                    print(f"{'📁 EMPTY':15} {dir_path}")
                    files_to_remove.append(full_path)
                else:
                    print(f"{'📁 HAS FILES':15} {dir_path:50} ({len(files_in_dir)} files)")
            except:
                print(f"{'❌ ERROR':15} {dir_path:50} (cannot access)")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 CLEANUP SUMMARY:")
    print(f"   Files to remove: {len(files_to_remove)}")
    print(f"   Space to save: {total_size_saved:,} bytes ({total_size_saved/1024:.1f} KB)")
    
    if not files_to_remove:
        print("✅ No files need to be removed. Project is already clean!")
        return
    
    # Execution
    if args.execute:
        print("\n🗑️  EXECUTING CLEANUP:")
        for file_path in files_to_remove:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    print(f"   ✅ Deleted file: {file_path.relative_to(project_root)}")
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    print(f"   ✅ Deleted directory: {file_path.relative_to(project_root)}")
            except Exception as e:
                print(f"   ❌ Error deleting {file_path.relative_to(project_root)}: {e}")
        
        print(f"\n✅ Cleanup completed! Saved {total_size_saved:,} bytes.")
        print("\n📋 RECOMMENDED NEXT STEPS:")
        print("   1. Update imports in remaining files to use centralized modules")
        print("   2. Test all Streamlit applications to ensure they still work")
        print("   3. Run: git add . && git commit -m 'Cleanup: Remove deprecated files'")
        
    else:
        print(f"\n⚠️  DRY RUN MODE - No files were actually deleted.")
        print(f"   To execute cleanup, run: python {Path(__file__).name} --execute")
        print(f"   To include notebooks: python {Path(__file__).name} --execute --include-notebooks")

if __name__ == "__main__":
    main()