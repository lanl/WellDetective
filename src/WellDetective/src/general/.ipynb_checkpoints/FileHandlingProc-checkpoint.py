# -*- coding: utf-8 -*-
"""
@author: James E. Lee
Contact: JamesEdLee@lanl.gov
Created: 21-May-2026
"""
#
import pandas as pd
from typing import Tuple, Dict

#

#
def detect_header_generic(filepath: str, max_lines: int = 50) -> Tuple[int, str]:
    """
    Generic function to detect header line and delimiter.
    Handles cases where header and data may use different delimiters.
    Returns r'\\s+' for whitespace-delimited files (tabs/spaces).
    
    Args:
        filepath: Path to file
        max_lines: Maximum lines to check
    
    Returns:
        Tuple of (header_line_number_0indexed, delimiter_for_data)
        Note: Returns r'\\s+' for whitespace-delimited files
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(max_lines)]
    
    # All common delimiters
    all_delimiters = ['\t', ';', '|', ',', ' ']
    
    best_match = None
    best_score = 0
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        
        # Try each delimiter for the current line (potential header)
        for header_delim in all_delimiters:
            if header_delim not in line:
                continue
            
            fields = [f.strip() for f in line.strip().split(header_delim) if f.strip()]
            
            # Check if this looks like a header (at least 3 columns)
            if len(fields) < 3:
                continue
            
            # Check if mostly non-numeric (header characteristic)
            header_numeric = sum(1 for f in fields if is_numeric(f))
            if header_numeric >= len(fields) * 0.5:
                continue  # Too many numbers to be a header
            
            # Now check if ANY subsequent line (within next 3) looks like data
            for j in range(i + 1, min(i + 4, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                
                # Try each delimiter for the data line (might be different from header!)
                for data_delim in all_delimiters:
                    if data_delim not in next_line:
                        continue
                    
                    next_fields = [f.strip() for f in next_line.split(data_delim) if f.strip()]
                    
                    # Need similar number of fields
                    if abs(len(fields) - len(next_fields)) > 2:
                        continue
                    
                    # Data should be mostly numeric
                    data_numeric = sum(1 for f in next_fields if is_numeric(f))
                    if data_numeric < len(next_fields) * 0.6:
                        continue  # Not enough numbers to be data
                    
                    # Calculate a score: prefer more fields, higher numeric ratio, and non-space delimiters
                    delimiter_bonus = 50 if data_delim != ' ' else 0
                    score = len(fields) + (data_numeric / len(next_fields)) * 100 + delimiter_bonus
                    
                    if score > best_score:
                        best_score = score
                        # KEY FIX: If header uses space and data uses tab/space, return \s+ pattern
                        if header_delim in [' ', '\t'] and data_delim in [' ', '\t']:
                            best_match = (i, r'\s+')
                        else:
                            best_match = (i, data_delim)
                        break  # Found good match for this header candidate
                
                if best_match and best_match[0] == i:
                    break  # Found match for this header, move to next potential header
    
    if best_match:
        return best_match
    
    return 0, ','
    
def is_numeric(s: str) -> bool:
    """Check if string is numeric."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False