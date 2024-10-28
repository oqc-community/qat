import typing
from typing import Annotated, Dict, List, Optional

from pydantic import PlainSerializer

from qat.utils.pydantic import WarnOnExtraFieldsModel

componentid = str


class HasId(WarnOnExtraFieldsModel):
    id: componentid


T = typing.TypeVar("T")
SerializeItemToRefs = PlainSerializer(lambda i: i.id, return_type=str, when_used="always")
Ref = Annotated[T, SerializeItemToRefs]

SerializeDictToRefs = PlainSerializer(
    lambda d: {k: v.id for (k, v) in d.items()}, return_type=dict, when_used="always"
)
RefDict = Annotated[Dict[componentid, T], SerializeDictToRefs]

SerializeListToRefs = PlainSerializer(
    lambda d: [v.id for v in d], return_type=list, when_used="always"
)
RefList = Annotated[List[T], SerializeListToRefs]

Context = Optional[Dict[componentid, HasId]]


def _get_field_type_and_container(cls, field):
    field_type = cls.model_fields[field].annotation
    container = typing.get_origin(field_type)
    field_cls = None
    if container is list:
        field_cls = typing.get_args(field_type)[0]
    elif container is dict:
        field_cls = typing.get_args(field_type)[1]
    elif container is None:
        field_cls = field_type
    else:
        raise Exception(f"Unknown container {container}")
    return field_cls, container


def _get_or_create(cls, context, data: str | dict | HasId):
    if issubclass(type(data), HasId):
        return data
    elif isinstance(data, str):
        return context.get(data)
    elif isinstance(data, dict):
        return cls(context=context, **data)


class HasId(WarnOnExtraFieldsModel):
    id: str

    def __init__(self, context=None, *args, **kw):
        kw = self.populate(context=context or {}, **kw)
        super().__init__(*args, **kw)

    @classmethod
    def populate(cls, context: Context = None, **data):
        """Populate a serialised data from a context

        The model fields are iterated the order they are declared
        any dict or list fields containing HasId items will be populated
        using the provided context.

        Sub models are contructed during population and the context is updated
        """
        for field in cls.model_fields:
            field_cls, container = _get_field_type_and_container(cls, field)
            if issubclass(field_cls, HasId):
                if container is dict and isinstance(data[field], dict):
                    data[field] = {
                        k: _get_or_create(field_cls, context, v)
                        for (k, v) in data[field].items()
                    }
                    new_context_items = data[field].values()
                elif container is list and isinstance(data[field], list):
                    data[field] = [
                        _get_or_create(field_cls, context, d) for d in data[field]
                    ]
                    new_context_items = data[field]
                elif container is None:
                    data[field] = _get_or_create(field_cls, context, data[field])
                    new_context_items = [data[field]]
                else:
                    found_type = type(data[field])
                    raise Exception(f"Container is {container}, found {found_type}")

                context.update({new.id: new for new in new_context_items})
        return data
