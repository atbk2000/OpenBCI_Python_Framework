from models.exception.framework_base_exception import FrameworkBaseException


class MissingParameterError(FrameworkBaseException):
    """
    Exception for missing parameter. This exception is used when a parameter is missing.
    """
    def __init__(self, module: str, name: str, parameter: str):
        super().__init__(exception_type='missing.parameter', module=module, name=name)
        self.message: str = f'{self.message}.{parameter}'
