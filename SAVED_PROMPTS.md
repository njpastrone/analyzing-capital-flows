# Implementing Changes from Research Pipeline Plan

"**Make a detailed implementation plan for [TASK DESCRIPTION] described in @[PLAN_DOCUMENT].md CRITICAL: Before planning, review:**
    @PROJECT_CONTEXT.md - Understand the problem context and constraints
    @CLAUDE.md - Check any critical warnings or project-specific guidelines
    @[OTHER_RELEVANT_DOCS] - Review any technical debt or existing issues
**Core Planning Requirements:**
    **What functionality do we need RIGHT NOW?**
        Identify the minimum viable implementation
        Focus on the most critical/risky component first
        List what can be deferred or excluded
        No unnecessary features or optimizations
    **Identify specific files and resources:**
        Which existing files will be read/modified/created?
        What data sources or APIs will be accessed?
        Which existing modules can be reused without modification?
        What are the input/output boundaries?
    **Write a complete implementation overview:**
        High-level step-by-step process
        Key technical decisions and trade-offs
        Directory/file structure to be created
        How existing code will be leveraged (not duplicated)
**List specific functions/components to create:**
        function_name() - 1-3 sentences about purpose and approach
        component_name - What it does and how it connects
        Include parameters and return types where critical
        Note any side effects or external dependencies
**Define validation/testing approach:**
        test_name() - 5-10 words about coverage
        How will correctness be verified?
        What are the success criteria?
        How will edge cases be handled?
**Critical constraints to remember:**
        DO NOT refactor or "improve" existing code
        DO NOT create parallel implementations that might diverge
        MUST show every intermediate calculation value
        Each notebook should be ~500 lines, NOT thousands
**Implementation milestones:**
        What proves the approach works? (Day 1)
        What are the minimum deliverables?
        What can be incrementally added?
        When is "done" actually done?
**Remember: Remember: We're creating transparent documentation of existing calculations, NOT rewriting the analysis system. The code already works - we're just making it traceable for academic review.**"

# Implementing Dashboard Consolidation Plan for CS1

        "CRITICAL IMPLEMENTATION TASK: Dashboard Consolidation - CS1 Proof of Concept Your Mission: Implement the dashboard consolidation plan outlined in @DASHBOARD_CONSOLIDATION_PLAN.md to reduce 12,800 lines of duplicated CS1 code down to ~3,300 lines while maintaining 100% functionality. BEFORE YOU START - Required Reading:
        Read @DASHBOARD_CONSOLIDATION_PLAN.md completely - this is your implementation blueprint
        Read @PROJECT_CONTEXT.md - understand why we're consolidating (NOT refactoring)
        Read @TECHNICAL_DEBT.md - understand the 4x duplication pattern we're fixing
        Review @CLAUDE.md warnings - DO NOT break working code
        CRITICAL CONSTRAINTS:
        This is CONSOLIDATION, not refactoring - combine duplicates using parameters
        Dashboard MUST remain fully functional at every step
        All calculations MUST remain identical (< 0.0001 difference)
        Archive original files, don't delete them
        Test each parameter combination before proceeding
        PHASE 1 IMPLEMENTATION STEPS (Hours 1-4):
        Create the new directory structure:
        mkdir -p src/dashboard/reports
        touch src/dashboard/reports/__init__.py
        Copy the base file:
        cp src/dashboard/full_reports/cs1_report_app.py src/dashboard/reports/cs1_report.py
        Modify cs1_report.py to add parameters:
        Change main function signature to: def main(data_type="full", output_mode="interactive", context="standalone")
        Add get_data_configuration(data_type) function to map parameters to data sources
        Add configure_ui_elements(output_mode) function to control UI rendering
        Update load_default_data() to use: analysis_type = 'winsorized' if data_type == 'winsorized' else 'full'
        Parameterize UI elements throughout the file:
        # Example pattern to apply:
        ui_config = configure_ui_elements(output_mode)

        if ui_config['use_expanders']:
        with st.expander("📋 Data and Methodology"):
                st.markdown(content)
        else:
        st.subheader("📋 Data and Methodology")
        st.markdown(content)

        if ui_config['show_download_buttons']:
        st.download_button(...)
        Compare with the 4 original versions to ensure you capture all differences:
        src/dashboard/full_reports/cs1_report_app.py - baseline
        src/dashboard/outlier_adjusted_reports/cs1_report_outlier_adjusted.py - check data loading difference
        src/dashboard/pdf_reports/cs1_report_app_pdf.py - check UI differences
        src/dashboard/pdf_reports_outlier_adjusted/cs1_report_outlier_adjusted_pdf.py - check both
        VALIDATION CHECKPOINT (Before Phase 2): Test all 4 parameter combinations in isolation:
        # Test these individually:
        main(data_type="full", output_mode="interactive")      # Should match full_reports version
        main(data_type="full", output_mode="pdf")              # Should match pdf_reports version
        main(data_type="winsorized", output_mode="interactive") # Should match outlier_adjusted version
        main(data_type="winsorized", output_mode="pdf")        # Should match pdf_outlier_adjusted version
        PHASE 2 IMPLEMENTATION (Hours 5-6):
        Update main_app.py imports:
        # Add new path
        sys.path.append(str(Path(__file__).parent / "reports"))

        # Change import
        from cs1_report import main as cs1_main
        Update all CS1 function calls in main_app.py:
        Search for case_study_1_main and replace with appropriate parameterized calls
        Ensure context parameter is passed through
        Test that main_app.py still runs without errors
        PHASE 3 VALIDATION (Hours 7-8):
        Visual comparison:
        Run original CS1 full report, take screenshot
        Run new consolidated CS1 with same parameters, take screenshot
        Compare visually - should be identical
        Numerical verification:
        Export F-test results from original to CSV
        Export F-test results from consolidated to CSV
        Diff the files - must be identical
        Document any discrepancies found
        SUCCESS CRITERIA: ✅ Single cs1_report.py file handles all 4 variants correctly ✅ All statistical results match exactly (< 0.0001 difference) ✅ UI renders identically in all modes ✅ main_app.py works with new consolidated version ✅ No console errors during execution IF SOMETHING BREAKS:
        DO NOT delete or modify the original files
        Document what went wrong
        Check if you missed a UI element difference or data loading variation
        Verify imports and function names match
        DELIVERABLES FOR DAY 1:
        Working src/dashboard/reports/cs1_report.py with all 4 modes functional
        Updated main_app.py using the consolidated version
        Test results showing all outputs match originals
        Brief summary of what was completed and any issues encountered
        IMPORTANT REMINDERS:
        You're CONSOLIDATING duplicates, not rewriting functionality
        The calculations are already correct - preserve them exactly
        If uncertain about a change, test it in isolation first
        Archive originals to src/dashboard/archive_20241203/ only AFTER everything works
        Start with Phase 1, Step 1. After each major step, briefly confirm completion before proceeding. If you encounter any blockers or uncertainties, stop and ask for clarification rather than making assumptions."