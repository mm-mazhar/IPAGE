import os
from typing import Any

import toml

# Load the pyproject.toml file
pyproject_path = "pyproject.toml"
requirements_path = "requirements.txt"

# Read the pyproject.toml file
pyproject_data: dict[str, Any] = toml.load(pyproject_path)

# Extract dependencies, ignoring the Python version
dependencies: dict[str, Any] = (
    pyproject_data.get("tool", {}).get("poetry", {}).get("dependencies", {})
)
dev_dependencies: dict[str, Any] = (
    pyproject_data.get("tool", {})
    .get("poetry", {})
    .get("group", {})
    .get("dev", {})
    .get("dependencies", {})
)

# Combine dependencies
all_dependencies: dict[str, Any] = {**dependencies, **dev_dependencies}

# Read existing requirements
existing_requirements = set()
if os.path.exists(requirements_path):
    with open(requirements_path, "r") as req_file:
        existing_requirements: set[str] = {line.strip() for line in req_file}

# Append to requirements.txt if not already present
with open(requirements_path, "a") as req_file:
    for package, version in all_dependencies.items():
        # Ignore the Python version and check if the package already exists
        if package != "python" and isinstance(version, str):
            formatted_dependency = f"{package}=={version.strip('^')}"
            if formatted_dependency not in existing_requirements:
                req_file.write(f"{formatted_dependency}\n")

# print(f"Dependencies have been appended to {requirements_path}.")
