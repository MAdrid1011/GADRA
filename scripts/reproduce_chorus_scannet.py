#!/usr/bin/env python3
"""Reproduce every fixed-manifest Chorus scene fragment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gadra.study import main

if __name__=='__main__':main('Chorus')
