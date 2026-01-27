"""
Regression Guard: Ensure no local binding of 'np' inside main() in climate_step2.
This prevents UnboundLocalError when 'import numpy as np' is at module level.
"""
import ast
import sys
from pathlib import Path
import pytest

def test_no_local_np_in_step2_main():
    repo_root = Path(__file__).parent.parent
    target_file = repo_root / "tools/climate_step2_aggregate_country_month.py"
    
    if not target_file.exists():
        pytest.skip(f"Tool not found: {target_file}")
        
    tree = ast.parse(target_file.read_text(encoding="utf-8"))
    
    # Find main function
    main_func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func = node
            break
            
    assert main_func is not None, "Could not find main() function in script"
    
    # Walk main() and check for assignments to 'np' or imports binding 'np'
    for node in ast.walk(main_func):
        # Check imports: import numpy as np
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == "np":
                    pytest.fail(f"Found local 'import {alias.name} as np' inside main() at line {node.lineno}")
                    
        # Check from imports: from ... import ... as np
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname == "np":
                    pytest.fail(f"Found local 'from ... import ... as np' inside main() at line {node.lineno}")
                    
        # Check assignments: np = ...
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "np":
                    pytest.fail(f"Found assignment to 'np' inside main() at line {node.lineno}")

if __name__ == "__main__":
    test_no_local_np_in_step2_main()
