"""
数据切片模块
用于将大型时间序列数据按时间窗口切片，支持强化学习训练
"""

import os
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm
import shutil


class DataSlicer:
    """数据切片器，将大型数据文件按时间窗口分割"""
    
    def __init__(self, slices_dir: str = 'data/slices'):
        """
        初始化数据切片器
        
        Args:
            slices_dir: 切片文件存储目录
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
        根据时间戳将数据分割成固定时长的片段
        
        Args:
            data_file: 原始数据文件路径 (.npz格式)
            pair_name: 交易对名称 (如 'xrpusdt')
            start_date: 开始日期 (如 20250717)
            hours_per_split: 每个片段的小时数
            force_recreate: 是否强制重新创建切片
            
        Returns:
            切片文件路径列表
        """
        # 创建基于数据文件的唯一目录
        file_hash = hashlib.md5(data_file.encode()).hexdigest()[:8]
        split_dir = os.path.join(
            self.slices_dir, 
            f"{pair_name}_{start_date}_{hours_per_split}h_{file_hash}"
        )
        
        # 检查是否已经存在分片
        if not force_recreate and os.path.exists(split_dir):
            existing_splits = sorted([f for f in os.listdir(split_dir) if f.endswith('.npz')])
            if len(existing_splits) > 0:
                print(f"📦 发现已存在的数据分片 ({len(existing_splits)}个)，直接使用...")
                split_files = [os.path.join(split_dir, f) for f in existing_splits]
                
                # 验证分片完整性
                if self._validate_splits(split_files):
                    return split_files
                else:
                    print("🔄 分片验证失败，重新创建...")
                    shutil.rmtree(split_dir)
        
        # 创建新的分片
        return self._create_splits(data_file, split_dir, hours_per_split)
    
    def _validate_splits(self, split_files: List[str], show_details: bool = True) -> bool:
        """
        验证分片文件的完整性
        
        Args:
            split_files: 分片文件路径列表
            show_details: 是否显示详细信息
            
        Returns:
            验证是否成功
        """
        try:
            total_records = 0
            for i, split_file in enumerate(split_files[:3] if show_details else split_files):
                data = np.load(split_file)
                data_length = len(data['data'])
                total_records += data_length
                
                if show_details and i < 3:
                    # 获取时间范围
                    if len(data['data']) > 0:
                        first_ts = data['data'][0]['exch_ts']
                        last_ts = data['data'][-1]['exch_ts']
                        duration_hours = (last_ts - first_ts) / 1e9 / 3600
                        print(f"   片段{i}: {data_length:,} 条记录 ({data_length/1e6:.1f}M) - {duration_hours:.1f}小时")
            
            if show_details and len(split_files) > 3:
                print(f"   ... 共{len(split_files)}个片段")
            
            return total_records > 0
        except Exception as e:
            print(f"⚠️ 分片验证失败: {e}")
            return False
    
    def _create_splits(
        self, 
        data_file: str, 
        split_dir: str, 
        hours_per_split: float
    ) -> List[str]:
        """
        创建新的数据分片
        
        Args:
            data_file: 原始数据文件
            split_dir: 分片保存目录
            hours_per_split: 每片段小时数
            
        Returns:
            创建的分片文件路径列表
        """
        os.makedirs(split_dir, exist_ok=True)
        print(f"📊 创建新的数据分片到: {split_dir}")
        
        # 加载原始数据
        print("   加载原始数据...")
        data = np.load(data_file)
        full_data = data['data']
        total_length = len(full_data)
        
        if total_length == 0:
            raise ValueError("数据文件为空")
        
        # 获取时间戳范围
        first_timestamp = full_data[0]['exch_ts']  # 纳秒
        last_timestamp = full_data[-1]['exch_ts']   # 纳秒
        total_duration_ns = last_timestamp - first_timestamp
        total_duration_hours = total_duration_ns / 1e9 / 3600
        
        print(f"   原始数据: {total_length:,} 条记录 ({total_length/1e6:.1f}M)")
        print(f"   时间范围: {total_duration_hours:.1f} 小时")
        print(f"   开始时间: {datetime.fromtimestamp(first_timestamp/1e9).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   结束时间: {datetime.fromtimestamp(last_timestamp/1e9).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 计算预期的片段数量
        expected_splits = int(np.ceil(total_duration_hours / hours_per_split))
        print(f"   预计生成 {expected_splits} 个片段（每{hours_per_split}小时）")
        
        # 按时间切片
        split_duration_ns = hours_per_split * 3600 * 1e9  # 转换为纳秒
        split_files = []
        
        print(f"   开始切片...")
        
        current_start_time = first_timestamp
        split_idx = 0
        
        # 使用tqdm显示切片进度
        with tqdm(total=expected_splits, desc="   切片进度", unit="片段") as pbar:
            while current_start_time < last_timestamp:
                current_end_time = current_start_time + split_duration_ns
                
                # 使用二分查找找到时间段内的数据
                start_idx = np.searchsorted(full_data['exch_ts'], current_start_time, side='left')
                end_idx = np.searchsorted(full_data['exch_ts'], current_end_time, side='left')
                
                if start_idx < end_idx:  # 确保有数据
                    split_data = full_data[start_idx:end_idx]
                    
                    # 保存片段
                    split_file = os.path.join(split_dir, f"split_{split_idx:03d}_{hours_per_split}h.npz")
                    np.savez_compressed(split_file, data=split_data)
                    split_files.append(split_file)
                    
                    # 计算实际时间长度
                    actual_duration = (split_data[-1]['exch_ts'] - split_data[0]['exch_ts']) / 1e9 / 3600
                    start_time_str = datetime.fromtimestamp(split_data[0]['exch_ts']/1e9).strftime('%H:%M:%S')
                    end_time_str = datetime.fromtimestamp(split_data[-1]['exch_ts']/1e9).strftime('%H:%M:%S')
                    
                    # 更新进度条描述
                    pbar.set_postfix({
                        '当前': f'片段{split_idx}',
                        '记录数': f'{len(split_data)/1e6:.1f}M',
                        '时长': f'{actual_duration:.1f}h',
                        '时间': f'{start_time_str}-{end_time_str}'
                    })
                    
                    split_idx += 1
                    pbar.update(1)
                
                # 移动到下一个时间段
                current_start_time = current_end_time
        
        # 显示切片结果摘要
        self._show_split_summary(split_files)
        
        print(f"\n✅ 数据分片完成，共{len(split_files)}个片段，保存在: {split_dir}")
        return split_files
    
    def _show_split_summary(self, split_files: List[str]):
        """
        显示切片结果摘要
        
        Args:
            split_files: 分片文件列表
        """
        print(f"\n   切片结果摘要:")
        for i, split_file in enumerate(split_files):
            if i < 3 or i >= len(split_files) - 1:  # 显示前3个和最后一个
                data = np.load(split_file)
                split_data = data['data']
                if len(split_data) > 0:
                    start_time = datetime.fromtimestamp(split_data[0]['exch_ts']/1e9).strftime('%Y-%m-%d %H:%M:%S')
                    end_time = datetime.fromtimestamp(split_data[-1]['exch_ts']/1e9).strftime('%H:%M:%S')
                    duration = (split_data[-1]['exch_ts'] - split_data[0]['exch_ts']) / 1e9 / 3600
                    print(f"     片段{i:03d}: {len(split_data):,} 条记录 ({len(split_data)/1e6:.1f}M) - "
                          f"{duration:.1f}小时 [{start_time} 至 {end_time}]")
            elif i == 3:
                print(f"     ...")
    
    def get_split_info(self, split_file: str) -> dict:
        """
        获取单个切片文件的信息
        
        Args:
            split_file: 切片文件路径
            
        Returns:
            包含切片信息的字典
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
        估算episode长度（步数）
        
        Args:
            data_file: 数据文件路径
            step_interval_ns: 步间隔（纳秒）
            
        Returns:
            估算的步数
        """
        try:
            # 快速加载数据以获取时间跨度
            data = np.load(data_file)
            if len(data['data']) == 0:
                return 1000  # 默认值
            
            # 获取时间跨度（纳秒）
            first_timestamp = data['data'][0]['exch_ts']
            last_timestamp = data['data'][-1]['exch_ts']
            time_span_ns = last_timestamp - first_timestamp
            
            # 计算步数：时间跨度 / 步间隔
            estimated_steps = int(time_span_ns / step_interval_ns)
            
            # 确保至少有一些步数
            return max(estimated_steps, 100)
        except Exception as e:
            print(f"估算步数失败: {e}")
            return 1000  # 默认估计值


# 便捷函数
def create_data_splits(
    data_file: str,
    pair_name: str,
    start_date: int,
    hours_per_split: float = 2.0,
    slices_dir: str = 'data/slices',
    force_recreate: bool = False
) -> List[str]:
    """
    创建数据切片的便捷函数
    
    Args:
        data_file: 原始数据文件路径
        pair_name: 交易对名称
        start_date: 开始日期
        hours_per_split: 每片段小时数
        slices_dir: 切片目录
        force_recreate: 是否强制重新创建
        
    Returns:
        切片文件路径列表
    """
    slicer = DataSlicer(slices_dir)
    return slicer.split_data_by_time(
        data_file, 
        pair_name, 
        start_date, 
        hours_per_split,
        force_recreate
    )