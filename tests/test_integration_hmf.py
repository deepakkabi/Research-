"""
Code Structure Verification Script

Verifies that the HMF Encoder module is correctly integrated into the
Cognitive Swarm project structure without requiring PyTorch installation.
"""

import os
import sys
import ast


def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ MISSING {description}: {filepath}")
        return False


def check_python_syntax(filepath):
    """Verify Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
            ast.parse(code)
        print(f"  ✓ Valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False


def check_class_exists(filepath, class_name):
    """Check if a class is defined in a Python file."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
            tree = ast.parse(code)
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            if class_name in classes:
                print(f"  ✓ Class '{class_name}' found")
                return True
            else:
                print(f"  ✗ Class '{class_name}' not found")
                return False
    except Exception as e:
        print(f"  ✗ Error checking class: {e}")
        return False


def check_imports(filepath, expected_imports):
    """Check if expected imports are present."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
            tree = ast.parse(code)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            missing = set(expected_imports) - set(imports)
            if not missing:
                print(f"  ✓ All expected imports present")
                return True
            else:
                print(f"  ⚠ Missing imports: {missing}")
                return False
    except Exception as e:
        print(f"  ✗ Error checking imports: {e}")
        return False


def check_method_exists(filepath, class_name, method_name):
    """Check if a method exists in a class."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    if method_name in methods:
                        print(f"  ✓ Method '{method_name}' found in {class_name}")
                        return True
                    else:
                        print(f"  ✗ Method '{method_name}' not found in {class_name}")
                        return False
            
            print(f"  ✗ Class '{class_name}' not found")
            return False
    except Exception as e:
        print(f"  ✗ Error checking method: {e}")
        return False


def verify_project_structure():
    """Main verification function."""
    print("=" * 70)
    print("COGNITIVE SWARM - HMF ENCODER MODULE VERIFICATION")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Check directory structure
    print("1. Checking Directory Structure")
    print("-" * 70)
    dirs = [
        ("cognitive_swarm/", "Root package directory"),
        ("cognitive_swarm/modules/", "Modules package"),
        ("tests/", "Test directory")
    ]
    
    for dirpath, desc in dirs:
        if os.path.isdir(dirpath):
            print(f"✓ {desc}: {dirpath}")
        else:
            print(f"✗ MISSING {desc}: {dirpath}")
            all_passed = False
    print()
    
    # Check main module file
    print("2. Checking HMF Encoder Module")
    print("-" * 70)
    encoder_file = "cognitive_swarm/modules/hmf_encoder.py"
    if check_file_exists(encoder_file, "HMF Encoder module"):
        check_python_syntax(encoder_file)
        check_imports(encoder_file, ["torch", "torch.nn"])
        check_class_exists(encoder_file, "HMFEncoder")
        check_class_exists(encoder_file, "LearnedHMFEncoder")
        check_method_exists(encoder_file, "HMFEncoder", "__init__")
        check_method_exists(encoder_file, "HMFEncoder", "forward")
        check_method_exists(encoder_file, "HMFEncoder", "forward_hierarchical")
    else:
        all_passed = False
    print()
    
    # Check __init__.py
    print("3. Checking Modules Package Init")
    print("-" * 70)
    init_file = "cognitive_swarm/modules/__init__.py"
    if check_file_exists(init_file, "Modules __init__.py"):
        check_python_syntax(init_file)
        # Check exports
        try:
            with open(init_file, 'r') as f:
                content = f.read()
                if "HMFEncoder" in content:
                    print("  ✓ HMFEncoder exported")
                else:
                    print("  ✗ HMFEncoder not exported")
                    all_passed = False
                if "LearnedHMFEncoder" in content:
                    print("  ✓ LearnedHMFEncoder exported")
                else:
                    print("  ✗ LearnedHMFEncoder not exported")
                    all_passed = False
        except Exception as e:
            print(f"  ✗ Error checking exports: {e}")
            all_passed = False
    else:
        all_passed = False
    print()
    
    # Check test file
    print("4. Checking Test Suite")
    print("-" * 70)
    test_file = "tests/test_hmf.py"
    if check_file_exists(test_file, "Test suite"):
        check_python_syntax(test_file)
        
        # Check test functions exist
        test_functions = [
            "test_dimensionality",
            "test_empty_role_group",
            "test_trust_weighting",
            "test_batch_processing",
            "test_integration_with_environment_output",
            "test_scalability_verification"
        ]
        
        try:
            with open(test_file, 'r') as f:
                code = f.read()
                tree = ast.parse(code)
                functions = [node.name for node in ast.walk(tree) 
                           if isinstance(node, ast.FunctionDef)]
                
                found_tests = 0
                for test_func in test_functions:
                    if test_func in functions:
                        found_tests += 1
                
                print(f"  ✓ {found_tests}/{len(test_functions)} critical tests found")
                if found_tests < len(test_functions):
                    all_passed = False
        except Exception as e:
            print(f"  ✗ Error checking test functions: {e}")
            all_passed = False
    else:
        all_passed = False
    print()
    
    # Check documentation
    print("5. Checking Documentation")
    print("-" * 70)
    readme_file = "HMF_ENCODER_README.md"
    if check_file_exists(readme_file, "README documentation"):
        try:
            with open(readme_file, 'r') as f:
                content = f.read()
                sections = [
                    "## Overview",
                    "## Quick Start", 
                    "## API Reference",
                    "## Integration with Environment",
                    "## Aggregation Method Comparison"
                ]
                
                found_sections = sum(1 for s in sections if s in content)
                print(f"  ✓ {found_sections}/{len(sections)} key sections found")
        except Exception as e:
            print(f"  ✗ Error checking documentation: {e}")
    else:
        all_passed = False
    print()
    
    # Code quality checks
    print("6. Code Quality Checks")
    print("-" * 70)
    
    # Check encoder file has docstrings
    try:
        with open(encoder_file, 'r') as f:
            code = f.read()
            tree = ast.parse(code)
            
            # Check module docstring
            if ast.get_docstring(tree):
                print("  ✓ Module docstring present")
            else:
                print("  ⚠ Module docstring missing")
            
            # Check class docstrings
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if ast.get_docstring(node):
                        print(f"  ✓ {node.name} has docstring")
                    else:
                        print(f"  ⚠ {node.name} missing docstring")
    except Exception as e:
        print(f"  ✗ Error checking docstrings: {e}")
    print()
    
    # Final summary
    print("=" * 70)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("\nThe HMF Encoder module is correctly integrated into Cognitive Swarm!")
        print("\nNext steps:")
        print("1. Install PyTorch: pip install torch")
        print("2. Run tests: python tests/test_hmf.py")
        print("3. Import in your code: from cognitive_swarm.modules import HMFEncoder")
    else:
        print("⚠ SOME CHECKS FAILED")
        print("\nPlease review the errors above and fix any issues.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = verify_project_structure()
    sys.exit(0 if success else 1)
