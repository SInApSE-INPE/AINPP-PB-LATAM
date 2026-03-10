import ast
import os
import sys

def get_imports(directory):
    modules = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    modules.add(alias.name.split(".")[0])
                            elif isinstance(node, ast.ImportFrom):
                                if node.module and node.level == 0:
                                    modules.add(node.module.split(".")[0])
                    except Exception as e:
                        pass
    return modules

def main():
    proj_dir = sys.argv[1]
    modules = set()
    modules.update(get_imports(os.path.join(proj_dir, "src")))
    modules.update(get_imports(os.path.join(proj_dir, "scripts")))
    modules.update(get_imports(os.path.join(proj_dir, "tests")))
    
    # Check root level .py files too
    for f in os.listdir(proj_dir):
        if f.endswith(".py"):
            try:
                with open(os.path.join(proj_dir, f), "r", encoding="utf-8") as file:
                    tree = ast.parse(file.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                modules.add(alias.name.split(".")[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and node.level == 0:
                                modules.add(node.module.split(".")[0])
            except:
                pass

    stdlib = set(sys.builtin_module_names) | {"sys", "os", "pathlib", "logging", "typing", "collections", "datetime", "math", "multiprocessing", "argparse", "warnings", "shutil", "traceback", "json"}
    
    external = sorted(list(modules - stdlib - {"ainpp"}))
    print(external)

if __name__ == "__main__":
    main()
