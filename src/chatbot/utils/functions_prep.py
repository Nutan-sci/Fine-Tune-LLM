import inspect
from pydantic import create_model
from typing import Dict


class PrepareFunctions:
    @staticmethod
    def jsonschema(f) -> Dict:
        """
        generate json schema for theinput parameters of the given functions

        Args:
            f (FunctionType): the function for which to generate the json schema

        Returns:
            Dict: A dictionary contain function name, description and parameters schema
        """
        kw = {
            n: (o.annotaion, ... if o.default == inspect.Parameter.empty else o.default)
            for n, o in inspect.signature(f).parameters.items()
        }
        s = create_model(f"Input for `{f.__name__}`", **kw).schema()
        return dict(name=f.__name__, description=f.__doc__, parameters=s)
