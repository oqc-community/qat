from typing import Annotated, get_args, get_origin

from qat.model.component import Component
from qat.model.hardware_model import QuantumHardwareModel


class QuantumHardwareModelBuilder:
    def __init__(self):
        self.model = QuantumHardwareModel()

    def add_component(self, component_type: Component, **kwargs):
        component = component_type(**kwargs)

        for field_name, field_type in self.model.__annotations__.items():
            if (
                self.model.model_fields[field_name].metadata
                and get_typevar_from_annotated(field_type) == component_type
            ):
                field_components = getattr(self.model, field_name)
                field_components.update({component.to_component_id(): component})

                self.model = self.model.model_copy(update={field_name: field_components})
                return

        raise Exception(f"Unknown hardware component type {component_type}.")


def get_typevar_from_annotated(type_name: str):
    """Function to retrieve the type assigned to T from a string representation of Annotated types."""
    annotated_type = eval(type_name)

    if get_origin(annotated_type) is Annotated:
        inner_type = get_args(annotated_type)[0]
        container_type = get_origin(inner_type)

        if container_type is dict:
            return get_args(inner_type)[1]

        elif container_type is list:
            return get_args(inner_type)[0]

        elif container_type is None:
            return inner_type

    raise ValueError("Provided Annotated type does not have the required structure.")
