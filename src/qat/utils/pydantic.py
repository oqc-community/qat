from pydantic import BaseModel, ConfigDict, model_validator

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class WarnOnExtraFieldsModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def validate(cls, fields):
        if unknown_fields := set(fields) - set(cls.model_fields):
            log.warning(
                f"Fields {unknown_fields}, which are not attributes of {cls.__name__}, will be ignored."
            )

        return fields
