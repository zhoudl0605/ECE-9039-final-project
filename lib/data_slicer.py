"""
Data slicing module
Used to slice large time series data by time windows, supporting reinforcement learning training
"""

import os
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm
import shutil


class DataSlicer:
    """Data slicer to split large data files by time windows"""
    
    def __init__(self, slices_dir: str = 'data/slices'):
        """
        Initialize data slicer
        
        Args:
            slices_dir: Directory to store slice files
        """
        self.slices_dir = slices_dir
        os.makedirs(slices_dir, exist_ok=True)
    
    def split_data_by_time(
        self, 
        data_file: str, 
        pair_name: str,
        start_date: int,
        hours_per_split: float = 2.0,
        force_recreate: bool = False
    ) -> List[str]:
        """
        Split data into fixed-duration segments based on timestamps
        
        Args:
            data_file: Path to original data file (.npz format)
            pair_name: Trading pair name (e.g. 'xrpusdt')
            start_date: Start date (e.g. 20250717)
            hours_per_split: Hours per segment
            force_recreate: Whether to force recreate slices
            
        Returns:
            List of slice file paths
        """
        file_hash = hashlib.md5(data_file.encode()).hexdigest()[:8]
        split_dir = os.path.join(
            self.slices_dir, 
            f"{pair_name}_{start_date}_{hours_per_split}h_{file_hash}"
        )
        
        # Check if slices already exist
        if not force_recreate and os.path.exists(split_dir):
            existing_splits = sorted([f for f in os.listdir(split_dir) if f.endswith('.npz')])
            if len(existing_splits) > 0:
                print(f"📦 Found existing data slices ({len(existing_splits)} files), using directly...")
                split_files = [os.path.join(split_dir, f) for f in existing_splits]
                
                # Validate slice integrity
                if self._validate_splits(split_files):
                    return split_files
                else:
                    print("🔄 Slice validation failed, recreating...")
                    shutil.rmtree(split_dir)
        
        # Create new slices
        return self._create_splits(data_file, split_dir, hours_per_split)
    
    def _validate_splits(self, split_files: List[str], show_details: bool = True) -> bool:
        """
        Validate slice file integrity
        
        Args:
            split_files: List of slice file paths
            show_details: Whether to show detailed information
            
        Returns:
            Whether validation succeeded
        """
        try:
            total_records = 0
            for i, split_file in enumerate(split_files[:3] if show_details else split_files):
                data = np.load(split_file)
                data_length = len(data['data'])
                total_records += data_length
                
                if show_details and i < 3:
                    if len(data['data']) > 0:
                        first_ts = data['data'][0]['exch_ts']
                        last_ts = data['data'][-1]['exch_ts']
                        duration_hours = (last_ts - first_ts) / 1e9 / 3600
                        print(f"   Segment {i}: {data_length:,} records ({data_length/1e6:.1f}M) - {duration_hours:.1f} hours")
            
            if show_details and len(split_files) > 3:
                print(f"   ... Total {len(split_files)} segments")
            
            return total_records > 0
        except Exception as e:
            print(f"⚠️ Slice validation failed: {e}")
            return False
    
    def _create_splits(
        self, 
        data_file: str, 
        split_dir: str, 
        hours_per_split: float
    ) -> List[str]:
        """
        Create new data slices
        
        Args:
            data_file: Original data file
            split_dir: Directory to save slices
            hours_per_split: Hours per segment
            
        Returns:
            List of created slice file paths
        """
        os.makedirs(split_dir, exist_ok=True)
        print(f"📊 Creating new data slices to: {split_dir}")
        
        # Load original data
        print("   Loading original data...")
        data = np.load(data_file)
        full_data = data['data']
        total_length = len(full_data)
        
        if total_length == 0:
            raise ValueError("Data file is empty")
        
        # Get timestamp range
        first_timestamp = full_data[0]['exch_ts']  # nanoseconds
        last_timestamp = full_data[-1]['exch_ts']   # nanoseconds
        total_duration_ns = last_timestamp - first_timestamp
        total_duration_hours = total_duration_ns / 1e9 / 3600
        
        print(f"   Original data: {total_length:,} records ({total_length/1e6:.1f}M)")
        print(f"   Time range: {total_duration_hours:.1f} hours")
        print(f"   Start time: {datetime.fromtimestamp(first_timestamp/1e9).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End time: {datetime.fromtimestamp(last_timestamp/1e9).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate expected number of segments
        expected_splits = int(np.ceil(total_duration_hours / hours_per_split))
        print(f"   Expected to generate {expected_splits} segments ({hours_per_split} hours each)")
        
        # Slice by time
        split_duration_ns = hours_per_split * 3600 * 1e9  # Convert to nanoseconds
        split_files = []
        
        print(f"   Starting slicing...")
        
        current_start_time = first_timestamp
        split_idx = 0
        
        # Show slicing progress with tqdm
        with tqdm(total=expected_splits, desc="   Slicing progress", unit="segments") as pbar:
            while current_start_time < last_timestamp:
                current_end_time = current_start_time + split_duration_ns
                
                # Binary search for time segment data
                start_idx = np.searchsorted(full_data['exch_ts'], current_start_time, side='left')
                end_idx = np.searchsorted(full_data['exch_ts'], current_end_time, side='left')
                
                if start_idx < end_idx:
                    split_data = full_data[start_idx:end_idx]
                    
                    # Save segment
                    split_file = os.path.join(split_dir, f"split_{split_idx:03d}_{hours_per_split}h.npz")
                    np.savez_compressed(split_file, data=split_data)
                    split_files.append(split_file)
                    
                    actual_duration = (split_data[-1]['exch_ts'] - split_data[0]['exch_ts']) / 1e9 / 3600
                    start_time_str = datetime.fromtimestamp(split_data[0]['exch_ts']/1e9).strftime('%H:%M:%S')
                    end_time_str = datetime.fromtimestamp(split_data[-1]['exch_ts']/1e9).strftime('%H:%M:%S')
                    
                    pbar.set_postfix({
                        'Current': f'Segment {split_idx}',
                        'Records': f'{len(split_data)/1e6:.1f}M',
                        'Duration': f'{actual_duration:.1f}h',
                        'Time': f'{start_time_str}-{end_time_str}'
                    })
                    
                    split_idx += 1
                    pbar.update(1)
                
                current_start_time = current_end_time
        
        # Show slice result summary
        self._show_split_summary(split_files)
        
        print(f"\n✅ Data slicing completed, {len(split_files)} segments saved in: {split_dir}")
        return split_files
    
    def _show_split_summary(self, split_files: List[str]):
        """
        Show slice result summary
        
        Args:
            split_files: List of slice files
        """
        print(f"\n   Slice result summary:")
        for i, split_file in enumerate(split_files):
            if i < 3 or i >= len(split_files) - 1:
                data = np.load(split_file)
                split_data = data['data']
                if len(split_data) > 0:
                    start_time = datetime.fromtimestamp(split_data[0]['exch_ts']/1e9).strftime('%Y-%m-%d %H:%M:%S')
                    end_time = datetime.fromtimestamp(split_data[-1]['exch_ts']/1e9).strftime('%H:%M:%S')
                    duration = (split_data[-1]['exch_ts'] - split_data[0]['exch_ts']) / 1e9 / 3600
                    print(f"     Segment {i:03d}: {len(split_data):,} records ({len(split_data)/1e6:.1f}M) - "
                          f"{duration:.1f} hours [{start_time} to {end_time}]")
            elif i == 3:
                print(f"     ...")
    
    def get_split_info(self, split_file: str) -> dict:
        """
        Get information about a single slice file
        
        Args:
            split_file: Slice file path
            
        Returns:
            Dictionary containing slice information
        """
        data = np.load(split_file)
        split_data = data['data']
        
        if len(split_data) == 0:
            return {
                'records': 0,
                'duration_hours': 0,
                'start_time': None,
                'end_time': None
            }
        
        first_ts = split_data[0]['exch_ts']
        last_ts = split_data[-1]['exch_ts']
        
        return {
            'records': len(split_data),
            'duration_hours': (last_ts - first_ts) / 1e9 / 3600,
            'start_time': datetime.fromtimestamp(first_ts/1e9),
            'end_time': datetime.fromtimestamp(last_ts/1e9),
            'first_timestamp_ns': first_ts,
            'last_timestamp_ns': last_ts
        }
    
    def estimate_episode_length(
        self, 
        data_file: str, 
        step_interval_ns: int = 1_000_000_000
    ) -> int:
        """
        Estimate episode length (number of steps)
        
        Args:
            data_file: Data file path
            step_interval_ns: Step interval (nanoseconds)
            
        Returns:
            Estimated number of steps
        """
        try:
            data = np.load(data_file)
            if len(data['data']) == 0:
                return 1000  # Default value
            
            first_timestamp = data['data'][0]['exch_ts']
            last_timestamp = data['data'][-1]['exch_ts']
            time_span_ns = last_timestamp - first_timestamp
            
            estimated_steps = int(time_span_ns / step_interval_ns)
            
            return max(estimated_steps, 100)
        except Exception as e:
            print(f"Failed to estimate steps: {e}")
            return 1000


# Convenience function
def create_data_splits(
    data_file: str,
    pair_name: str,
    start_date: int,
    hours_per_split: float = 2.0,
    slices_dir: str = 'data/slices',
    force_recreate: bool = False
) -> List[str]:
    """
    Convenience function to create data slices
    
    Args:
        data_file: Original data file path
        pair_name: Trading pair name
        start_date: Start date
        hours_per_split: Hours per segment
        slices_dir: Slice directory
        force_recreate: Whether to force recreate
        
    Returns:
        List of slice file paths
    """
    slicer = DataSlicer(slices_dir)
    return slicer.split_data_by_time(
        data_file, 
        pair_name, 
        start_date, 
        hours_per_split,
        force_recreate
    )