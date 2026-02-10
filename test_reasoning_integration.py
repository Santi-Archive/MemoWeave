#!/usr/bin/env python3
"""
Test script to verify reasoning graph integration in character.py
Tests both backward compatibility and enhanced functionality.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from character import load_reasoning_graph, format_reasoning_graph, build_prompt

def test_load_reasoning_graph_missing():
    """Test: Loading reasoning graph when file doesn't exist (should return None)"""
    print("Test 1: Load reasoning graph when missing...")
    result = load_reasoning_graph("nonexistent_directory")
    if result is None:
        print("✅ PASS: Returns None when file missing")
        return True
    else:
        print("❌ FAIL: Should return None")
        return False

def test_format_reasoning_graph_none():
    """Test: Formatting None should return empty string"""
    print("\nTest 2: Format None reasoning graph...")
    result = format_reasoning_graph(None)
    if result == "":
        print("✅ PASS: Returns empty string for None")
        return True
    else:
        print("❌ FAIL: Should return empty string")
        return False

def test_format_reasoning_graph_valid():
    """Test: Formatting valid reasoning graph"""
    print("\nTest 3: Format valid reasoning graph...")
    test_graph = {
        "temporal_relations": [
            {"from_event": "E1", "to_event": "E2", "relation": "BEFORE"}
        ],
        "causal_relations": [
            {"from_event": "E1", "to_event": "E3", "relation": "CAUSES"}
        ]
    }
    result = format_reasoning_graph(test_graph)
    
    if "TEMPORAL & CAUSAL RELATIONS" in result and "E1" in result and "BEFORE" in result:
        print("✅ PASS: Formats reasoning graph correctly")
        print(f"Output preview:\n{result[:200]}...")
        return True
    else:
        print("❌ FAIL: Formatting error")
        return False

def test_build_prompt_backward_compatible():
    """Test: build_prompt without reasoning graph (backward compatibility)"""
    print("\nTest 4: Build prompt without reasoning graph...")
    chapters = {
        "1": ["- Event A (actor: John, target: door, location: room)"]
    }
    result = build_prompt(chapters)
    
    if "Chapter 1:" in result and "Event A" in result and "TEMPORAL" not in result:
        print("✅ PASS: Backward compatible - works without reasoning graph")
        return True
    else:
        print("❌ FAIL: Backward compatibility issue")
        return False

def test_build_prompt_with_reasoning():
    """Test: build_prompt with reasoning graph (enhanced functionality)"""
    print("\nTest 5: Build prompt with reasoning graph...")
    chapters = {
        "1": ["- Event A (actor: John, target: door, location: room)"]
    }
    test_graph = {
        "temporal_relations": [
            {"from_event": "E1", "to_event": "E2", "relation": "BEFORE"}
        ]
    }
    result = build_prompt(chapters, test_graph)
    
    if "Chapter 1:" in result and "TEMPORAL & CAUSAL RELATIONS" in result:
        print("✅ PASS: Enhanced mode - includes reasoning graph")
        return True
    else:
        print("❌ FAIL: Enhanced mode not working")
        return False

def main():
    print("=" * 60)
    print("Reasoning Graph Integration - Test Suite")
    print("=" * 60)
    
    tests = [
        test_load_reasoning_graph_missing,
        test_format_reasoning_graph_none,
        test_format_reasoning_graph_valid,
        test_build_prompt_backward_compatible,
        test_build_prompt_with_reasoning
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED - Implementation is safe and working!")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED - Review implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
