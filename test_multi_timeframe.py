#!/usr/bin/env python3
"""
Test script for multi-timeframe functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.multi_timeframe import get_multi_timeframe_features, get_timeframe_summary
from config import Config
from utils.logger import get_logger

logger = get_logger('test_mtf', log_file='logs/test_mtf.log')

def test_multi_timeframe():
    """Test multi-timeframe functionality"""
    print("Testing Multi-Timeframe Analysis...")
    print("=" * 50)
    
    # Test with USDJPY
    symbol = 'USDJPY'
    print(f"\n1. Testing timeframe summary for {symbol}")
    try:
        summary = get_timeframe_summary(symbol, lookback_days=90)
        print("Timeframe Summary:")
        for tf, data in summary.items():
            print(f"  {tf}: {data['rows']} rows, Price: {data['current_price']:.5f}")
    except Exception as e:
        print(f"Error getting timeframe summary: {e}")
    
    print(f"\n2. Testing multi-timeframe features for {symbol}")
    try:
        features_df = get_multi_timeframe_features(symbol, lookback_days=60)
        if features_df is not None and not features_df.empty:
            print(f"Success! Features shape: {features_df.shape}")
            print(f"Columns: {list(features_df.columns)}")
            
            # Show some multi-timeframe specific columns
            mtf_cols = [col for col in features_df.columns if any(tf in col for tf in ['15m', '4h'])]
            print(f"\nMulti-timeframe columns ({len(mtf_cols)}):")
            for col in mtf_cols[:10]:  # Show first 10
                print(f"  {col}")
            if len(mtf_cols) > 10:
                print(f"  ... and {len(mtf_cols) - 10} more")
        else:
            print("No features generated")
    except Exception as e:
        print(f"Error getting multi-timeframe features: {e}")
    
    # Test with BTCUSD
    symbol = 'BTCUSD'
    print(f"\n3. Testing multi-timeframe features for {symbol}")
    try:
        features_df = get_multi_timeframe_features(symbol, lookback_days=60)
        if features_df is not None and not features_df.empty:
            print(f"Success! Features shape: {features_df.shape}")
            mtf_cols = [col for col in features_df.columns if any(tf in col for tf in ['15m', '4h'])]
            print(f"Multi-timeframe columns: {len(mtf_cols)}")
        else:
            print("No features generated")
    except Exception as e:
        print(f"Error getting multi-timeframe features: {e}")
    
    print("\n" + "=" * 50)
    print("Multi-timeframe test completed!")

if __name__ == "__main__":
    test_multi_timeframe() 