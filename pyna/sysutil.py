import importlib.util
import sys

def if_package_installed(package_name:str) -> bool:
    """check whether a specified package has been installed. The script is from Christopher, [StackOverflow: Check if Python Package is installed](https://stackoverflow.com/q/1051254/12486177)

    Args:
        package_name (str): e.g., "numpy", "cupy" or etc.

    Returns:
        bool: whether the package has been installed.
    """
    if package_name in sys.modules:
        # print(f"{package_name!r} already in sys.modules")
        return True
    elif (spec := importlib.util.find_spec(package_name)) is not None:
        # If you choose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
        # print(f"{package_name!r} has been imported")
        return True 
    else:
        # print(f"can't find the {package_name!r} module")
        return False

def check_lambdify_package_available(lambda_type:str):
    if lambda_type == "numpy":
        if not if_package_installed("numpy"):
            raise ImportError("The lambdifying requires numpy package.")
    elif lambda_type == "cupy":
        if not if_package_installed("cupy"):
            raise ImportError("The lambdifying requires cupy package.")
    else:
        raise NotImplementedError()