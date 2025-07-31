"""
Data Loader for Cleaned Capital Flows Datasets
Provides easy access to the pre-cleaned datasets in updated_data/Clean/
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Union

class CleanDataLoader:
    """Loads pre-cleaned capital flows datasets from updated_data/Clean/"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize with data directory path"""
        if data_dir is None:
            # Default to updated_data/Clean relative to project root
            self.data_dir = Path(__file__).parent.parent.parent / "updated_data" / "Clean"
        else:
            self.data_dir = Path(data_dir)
    
    def load_comprehensive_labeled(self, **kwargs) -> pd.DataFrame:
        """
        Load the comprehensive labeled dataset (% of GDP format)
        
        Returns:
            DataFrame with CS1_GROUP, CS2_GROUP, CS3_GROUP columns for case study filtering
        """
        file_path = self.data_dir / "comprehensive_df_PGDP_labeled.csv "
        return pd.read_csv(file_path, **kwargs)
    
    def load_comprehensive_pgdp(self, **kwargs) -> pd.DataFrame:
        """Load comprehensive dataset in % of GDP format"""
        file_path = self.data_dir / "comprehensive_df_PGDP.csv "
        return pd.read_csv(file_path, **kwargs)
    
    def load_comprehensive_usd(self, **kwargs) -> pd.DataFrame:
        """Load comprehensive dataset in USD format"""
        file_path = self.data_dir / "comprehensive_df_USD.csv "
        return pd.read_csv(file_path, **kwargs)
    
    def load_case_study_data(self, case_study: int, **kwargs) -> pd.DataFrame:
        """
        Load specific case study data
        
        Args:
            case_study: 1, 2, or 3/4
        """
        file_map = {
            1: "case_one_data_USD.csv",
            2: "case_two_data_USD.csv", 
            3: "case_three_four_data_USD.csv",
            4: "case_three_four_data_USD.csv"
        }
        
        if case_study not in file_map:
            raise ValueError(f"Case study {case_study} not available. Use 1, 2, 3, or 4")
        
        file_path = self.data_dir / file_map[case_study]
        return pd.read_csv(file_path, **kwargs)
    
    def filter_by_case_study(self, data: pd.DataFrame, case_study: int, 
                           group: Optional[str] = None) -> pd.DataFrame:
        """
        Filter labeled data by case study and optional group
        
        Args:
            data: DataFrame with case study group columns
            case_study: 1, 2, or 3
            group: Optional specific group within case study
            
        Returns:
            Filtered DataFrame
        """
        group_col = f"CS{case_study}_GROUP"
        
        if group_col not in data.columns:
            raise ValueError(f"Column {group_col} not found in data")
        
        # Filter out NaN values for the case study
        filtered = data[data[group_col].notna()].copy()
        
        # Further filter by specific group if provided
        if group is not None:
            filtered = filtered[filtered[group_col] == group].copy()
        
        return filtered
    
    def get_available_indicators(self, data: pd.DataFrame, 
                               format_type: str = "PGDP") -> List[str]:
        """
        Get list of available indicators
        
        Args:
            data: DataFrame to examine
            format_type: "PGDP" for % of GDP indicators, "USD" for USD indicators
            
        Returns:
            List of indicator column names
        """
        if format_type == "PGDP":
            return [col for col in data.columns if col.endswith('_PGDP')]
        elif format_type == "USD":
            return [col for col in data.columns if col.endswith('_USD')]
        else:
            raise ValueError("format_type must be 'PGDP' or 'USD'")
    
    def get_case_study_info(self, data: pd.DataFrame) -> dict:
        """
        Get information about case study groups in the data
        
        Returns:
            Dictionary with case study group counts
        """
        info = {}
        
        for case_study in [1, 2, 3]:
            group_col = f"CS{case_study}_GROUP"
            if group_col in data.columns:
                info[f"Case Study {case_study}"] = data[group_col].value_counts().to_dict()
        
        return info

# Convenience functions for quick access
def load_labeled_data(**kwargs) -> pd.DataFrame:
    """Quick access to labeled comprehensive dataset"""
    loader = CleanDataLoader()
    return loader.load_comprehensive_labeled(**kwargs)

def filter_case_study(data: pd.DataFrame, case_study: int, group: Optional[str] = None) -> pd.DataFrame:
    """Quick filter by case study"""
    loader = CleanDataLoader()
    return loader.filter_by_case_study(data, case_study, group)