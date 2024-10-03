# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import atexit
import logging
import os
import shutil
import sys
from datetime import date, datetime
from enum import Enum
from logging.config import dictConfig
from pathlib import Path
from typing import IO, List, Union

from compiler_config.serialiser import json_dump, json_load
from numpy import savetxt

# Formatted to "[INFO] 2020-08-25 19:54:28,216 (module_name.function_name:line_number) - message"
default_logger_format = "[%(levelname)s] %(asctime)s - %(name)s - (%(module)s.%(funcName)s:%(lineno)d) - %(message)s"
json_format = (
    '{"level": "%(levelname)s", "time": "%(asctime)s",'
    '"module name": "%(module)s", "function name": "%(funcName)s", "line number": %(lineno)d,'
    '"message": "%(message)s"},'
)


class LoggerLevel(Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    OUTPUT = logging.NOTSET + 2
    CODE = logging.NOTSET + 1
    NOTSET = logging.NOTSET

    def __repr__(self):
        return self.name


class BasicLogger(logging.Logger):
    """
    The basic logger class that should be used. Upon setup, this is provided to the
    built-in logging by calling ``logging.setLoggerClass``. This way, every new logger
    created will be of this class. This allows us to define custom fields and functions
    that we want to use with our loggers. This class should not be instantiated
    separately, only call ``logging.getLogger("qat.purr.some_name")``, and this will return
    an instance of :class:`BasicLogger`.
    """

    def __init__(self, name: str, _log_folder: "LogFolder" = None):
        logging.Logger.__init__(self, name)
        self.setLevel(logging.INFO)
        self.log_folder = _log_folder or LogFolder()

    @property
    def logs_path(self):
        return self.log_folder.folder_path

    def output(
        self,
        data,
        *args,
        cell_type: str = None,
        fit_type: str = None,
        msg: str = "",
        section_level=1,
    ):
        """
        Displays the result of some experiment or computation. This logging function is
        PuRR specific.

        The output depends on the :paramref:`cell_type` parameter:

        - ``table`` - it creates a table in markdown syntax. The data needs to be a list
          of lists of strings, where the first list will be the header of the table, and
          then each list represents a row.
        - ``fit`` - it displays the Fit function in LaTeX syntax. It also requires the
          :paramref:`fit_type` parameter
        - ``new_section`` - creates a new section in markdown syntax. The title will be
          the string provided by :paramref:`data`

        :param data: The data to output based on the rest of the parameters.
        :param cell_type: It can have multiple values: ``table``, ``fit``,
            ``new_section``
        :param fit_type: Required when :paramref:`cell_type` is ``fit``. It can be one
            of the following types: ``SINUSOID``, ``SINUSOID_EXPONENTIAL``,
            ``POLYNOMIAL``, ``EXPONENTIAL``, ``LORENTZIAN``
        :param msg: Message to insert in front of the table if :paramref:`cell_type` is
            ``table``
        :param section_level: The level of the new section in markdown, if
            :paramref:`cell_type` is ``new_section``
        """
        logging.Logger.log(
            self,
            level=LoggerLevel.OUTPUT.value,
            msg=data,
            *args,
            extra={
                "cell_type": cell_type,
                "fit_type": fit_type,
                "note": msg,
                "section_level": section_level,
            },
        )

    def code(self, source: List[str]):
        """
        Outputs a section of executable code (used for Jupyter files). This logging
        function is PuRR specific.

        :param source: The list of code lines as strings
        """
        logging.Logger.log(self, level=LoggerLevel.CODE.value, msg="\n".join(source))

    def save_object(self, obj, name: str, numpy_arr: bool = False):
        """
        Serializes the specified object. This logging function is PuRR specific.

        :param obj: The object to be serialized.
        :param name: A name must be also provided, which will be used as the file name
                     and in case of Jupyter loggers, also as the name of the variable
                     which will contain the loaded object.
        """
        if numpy_arr:
            code = [
                "# load the measured data from the file",
                "from numpy import loadtxt",
                f"{name} = loadtxt(r'{name}.txt')",
            ]
            logging.Logger.log(self, level=LoggerLevel.CODE.value, msg="\n".join(code))
            return
        file_name = f"{name}.json"
        file_path = os.path.join(self.logs_path, file_name)
        with open(file_path, "w") as f:
            json_dump(obj, f, indent=4, ensure_ascii=False)

        # After serializing the object above, a code snippet will be added to the
        # Jupyter notebook to load the object from the JSON, and save it to a variable.
        # This will allow the user to actively analyse the saved object after running
        # the Jupyter script.
        code = [
            "from qat.purr.logger import *",
            f"{name} = load_data('{os.path.relpath(file_path, self.logs_path)}')",
        ]
        logging.Logger.log(self, level=LoggerLevel.CODE.value, msg="\n".join(code))

    def close(self):
        """Closes this logger, cleans up the file handles and appropriate folder."""
        for handler in self.handlers:
            try:
                handler.close()
            except Exception as e:
                print(f"Logger handler failed to close cleanly. Message: {e}")

        if self.log_folder is not None:
            try:
                self.log_folder.close()
            except Exception as e:
                print(f"Log folder failed to close cleanly. Message: {e}")

    _dummy_log = logging.makeLogRecord({})
    record_override_key = "$_enable_record_override"

    def makeRecord(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra: dict = None,
        sinfo=None,
    ):
        """
        Override that allows us to override record values via the extras dictionary.
        Initially built to allow for printing out messages that look like they come from
        different places in the source code than where the logger was called.
        """
        if extra is None:
            extra = {}

        if extra.pop(self.record_override_key, False):
            # Strip off values that would conflict with the makeRecord validation then
            # just apply them afterwards.
            overwriting_values = {
                key: value
                for key, value in extra.items()
                if key in self._dummy_log.__dict__
            }
            for key in overwriting_values:
                del extra[key]

            # If we have overrides for the function values, just apply them.
            if "fn" in extra:
                fn = extra.pop("fn")

            if "lno" in extra:
                lno = extra.pop("lno")

            if "func" in extra:
                func = extra.pop("func")
        else:
            overwriting_values = {}

        record = super().makeRecord(
            name, level, fn, lno, msg, args, exc_info, func, extra, sinfo
        )
        for key, value in overwriting_values.items():
            record.__dict__[key] = value

        return record

    def __enter__(self):
        pass

    def __exit__(self, exc, value, tb):
        self.close()

    def __del__(self):
        self.close()


class ConsoleLoggerHandler(logging.StreamHandler):
    """
    Basic console handler for the logger. It defaults to stdout.
    """

    def __init__(self, stream: IO = sys.stdout):
        super().__init__(stream)
        self.setFormatter(logging.Formatter(default_logger_format))

    def __repr__(self):
        return "Console logger handler"


class FileLoggerHandler(logging.FileHandler):
    """
    Basic file handler for the logger. A file path must be provided. The log file is
    created with a delay, so the stream is None until the first emit. This also allows
    to write some initial stuff to the log file when creating it.
    """

    def __init__(self, file_path: str):
        super().__init__(os.path.abspath(file_path), mode="w", delay=True)
        self.setFormatter(logging.Formatter(default_logger_format))

    def emit(self, record):
        if self.stream is None:
            os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)
            self.stream = self._open()
            self.create_initial_file()
        logging.FileHandler.emit(self, record)

    def create_initial_file(self):
        """
        Implement this method in the derived class to insert some initial text in the
        log file. Use emit and flush while writing directly to the stream.
        """
        pass

    def __repr__(self):
        return "File logger handler (path: %s)" % self.baseFilename


class JsonHandler(FileLoggerHandler):
    """
    The JSON file handler needed for logging in JSON format. In
    :class:`logging.FileHandler`, at each time something is written to the file, the
    emit function is called. By overriding the method, the JSON format can be ensured at
    each writing.
    """

    def __init__(self, file_path):
        super().__init__(file_path)
        self.file_terminator = "{}]}" + self.terminator

    def create_initial_file(self):
        pass

    def emit(self, record):
        # TODO: Unicode problem
        if self.stream is None:
            os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)
            self.stream = self._open()
            self.create_initial_file()
        self.stream.seek(0, os.SEEK_END)
        self.stream.seek(self.stream.tell() - len(self.file_terminator) - 1, os.SEEK_SET)
        FileLoggerHandler.emit(self, record)
        self.stream.write(self.file_terminator)
        self.flush()


class JsonLoggerHandler(JsonHandler):
    """
    The basic JSON file handler logger. It is intended to generate the same output as
    :class:`FileLoggerHandler`, but in JSON format.
    """

    def __init__(self, file_path):
        extension = Path(file_path).suffix
        if not extension == ".json":
            raise ValueError("Only JSON file paths are accepted!")
        super().__init__(file_path)
        self.setFormatter(logging.Formatter(json_format))

    def create_initial_file(self):
        self.stream.write(
            "{" + f'{os.linesep}"entries": [{os.linesep}' + "{}]}" + self.terminator
        )
        self.flush()

    def __repr__(self):
        return "JSON console logger handler (path: %s)" % self.baseFilename


class CompositeLogger(BasicLogger):
    """
    The default logger class of PuRR. It is intended to store all the loggers in a list,
    and when logging, the functions from here should be called, which iterate through
    the list and apply the logging to each logger separately. This way, only one
    function needs to be called when logging, and it is ensured, that all the enabled
    loggers will log the message.
    """

    def __init__(
        self,
        loggers_or_names: List[Union[str, logging.Logger]] = None,
        _log_folder=None,
    ):
        """Creates the list of loggers on which the logging functions will iterate

        :param logger_names: List of loggers by their names
            (e.g. ``["qat.purr.json", "qat.purr.file"]``) or actual logger instances.
        """
        super().__init__("default", _log_folder)

        self.loggers = []
        if loggers_or_names is None:
            loggers_or_names = []

        # Add root to our composite so consumers can also hook into our logs.
        root = logging.getLogger()
        loggers_or_names.append(root)
        self.add_loggers(loggers_or_names)

        # Set the root to the lowest non-custom log level activated.
        root.setLevel(
            min([val.level for val in self.loggers if (float(val.level / 10)).is_integer()])
        )

    def add_loggers(self, loggers_or_names: List[Union[str, logging.Logger]] = ()):
        if loggers_or_names is not None:
            for val in loggers_or_names:
                if isinstance(val, str):
                    self.loggers.append(logging.getLogger(val))
                elif isinstance(val, logging.Logger):
                    self.loggers.append(val)

    def _add_stack_levels(self, kwargs):
        """
        Due to the way the loggers work, we need to go back up the stack a few calls to
        get the real caller.
        """
        kwargs["stacklevel"] = kwargs.get("stacklevel", 0) + 2
        return kwargs

    def info(self, msg: str, *args, **kwargs):
        kwargs = self._add_stack_levels(kwargs)
        for logger in self.loggers:
            logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        kwargs = self._add_stack_levels(kwargs)
        for logger in self.loggers:
            logger.debug(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        kwargs = self._add_stack_levels(kwargs)
        for logger in self.loggers:
            logger.error(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        kwargs = self._add_stack_levels(kwargs)
        for logger in self.loggers:
            logger.warning(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        kwargs = self._add_stack_levels(kwargs)
        for logger in self.loggers:
            logger.critical(msg, *args, **kwargs)

    def log(self, level, msg: str, *args, **kwargs):
        kwargs = self._add_stack_levels(kwargs)
        if isinstance(level, LoggerLevel):
            level = level.value
        for logger in self.loggers:
            logger.log(level, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        kwargs = self._add_stack_levels(kwargs)
        for logger in self.loggers:
            logger.exception(msg, *args, **kwargs)

    def output(self, data, **kwargs):
        for logger in self.loggers:
            if isinstance(logger, BasicLogger):
                logger.output(data, **kwargs)

    def code(self, source: List[str]):
        for logger in self.loggers:
            if isinstance(logger, BasicLogger):
                logger.code(source)

    def save_object(self, obj, name: str, numpy_arr: bool = False, overwrite=False):
        """
        Iterates through the list of loggers and calls the corresponding save_object
        implementations for each of them. This is the method that should be called from
        the code.
        """
        if numpy_arr:
            if not overwrite:
                i = 2
                original_name = name
                while os.path.exists(
                    os.path.join(self.log_folder.folder_path, f"{name}.txt")
                ):
                    name = f"{original_name}_{i}"
                    i += 1
            file_path = os.path.join(self.log_folder.folder_path, f"{name}.txt")
            savetxt(file_path, obj)
        for logger in self.loggers:
            if isinstance(logger, BasicLogger):
                logger.save_object(obj, name, numpy_arr=numpy_arr)
        return name

    def close(self):
        super(CompositeLogger, self).close()
        for logger in self.loggers:
            if isinstance(logger, BasicLogger):
                logger.close()


class LogFolder:
    """
    It is the main log folder in which all the log files are saved. It can be configured
    multiple ways, like the base folder path, which can be a specified path of the disk
    or ``None`` to save the logs in the system temporary folder. The
    :paramref:`labber_style` specifies whether to create a Labber-style folder hierarchy
    for the logs or not. If not, depending on the :paramref:`folder_name` parameter, it
    will either create random folders for each run, or if :paramref:`folder_name` is not
    ``None``, it will create a sub-folder specified by :paramref:`folder_name`. Also, a
    :paramref:`prefix` and :paramref:`suffix` can be specified to append the created log
    folder (if :paramref:`labber_style` is not True).
    """

    def __init__(
        self,
        base_folder_path: str = None,
        labber_style: bool = None,
        cleanup: bool = None,
        folder_name: str = None,
        prefix: str = None,
        suffix: str = None,
    ):
        """
        The constructor for the LogFolder. It can be configured by the parameters. If
        the parameters are not provided, the default variables are used from the front
        of the file (which can be also set by importing a configuration).

        :param base_folder_path: Specifies the base directory, where the new log folder
            will be created. If it is ``None``, then it is set to the default value
            ``default_logger_base_directory``, which is set by the imported
            configuration file, otherwise it is defined at the top of this module. If
            the default value is ``None``, the log folder will be created in the
            system's TMP folder.
        :param labber_style: If it is true, it will create a labber hierarchy log
            folder.
        :param cleanup: If it is true, it will remove the log folder together with the
            logs at the end of execution.
        :param folder_name: If :paramref:`labber_style` is false, then
            :paramref:`folder_name` will be the name of the new log folder instead of
            generating a random one.
        :param prefix: It appends to the front of the generated folder name
        :param suffix: It appends to the end of the generated folder name
        """
        self.starting_date = date.today()
        self.needs_cleanup: bool = cleanup

        if base_folder_path is None:
            base_folder_path = "logs"

        self.base_folder_path = os.path.abspath(base_folder_path)
        os.makedirs(self.base_folder_path, exist_ok=True)

        folder_name = folder_name or ""
        if prefix is not None:
            folder_name = f"{prefix}_{folder_name}"
        if suffix is not None:
            folder_name = f"{folder_name}_{suffix}"
        if labber_style:
            self.folder_path = self.create_sub_folder_labber_style(
                base_folder_path, folder_name=folder_name
            )
        else:
            self.folder_path = os.path.join(base_folder_path, folder_name)
            os.makedirs(self.folder_path, exist_ok=True)

        self.folder_path = os.path.abspath(self.folder_path)

    def get_log_file_path(self, file_name: str = "log", over_write=True):
        file_path = os.path.join(self.folder_path, file_name)
        if not over_write:
            if os.path.exists(file_path):
                raise OSError("File already exists!")
        return file_path

    def create_sub_folder_labber_style(self, base_folder, folder_name: str = None):
        if folder_name is None or folder_name == "":
            folder_name = ""
        else:
            folder_name = folder_name + "."
        now = datetime.now()
        folder_name = folder_name + f"{now.hour:02d}.{now.minute:02d}.{now.second:02d}"
        main_folder_path = self.get_main_folder_path_labber_style(base_folder)
        original_name = folder_name
        i = 2
        while os.path.exists(os.path.join(main_folder_path, folder_name)):
            folder_name = f"{original_name}_{i}"
            i += 1
        final_sub_folder_path = os.path.join(main_folder_path, folder_name)
        os.makedirs(final_sub_folder_path, exist_ok=True)
        return final_sub_folder_path

    @staticmethod
    def get_main_folder_path_labber_style(base_folder):
        now = datetime.now()
        year, month, day = now.year, now.month, now.day
        main_folder_path = os.path.join(
            base_folder, f"{year:04d}", f"{month:02d}", f"Data_{month:02d}{day:02d}"
        )
        return main_folder_path

    def close(self):
        if self.folder_path is not None and os.path.isdir(self.folder_path):
            if self.needs_cleanup:
                shutil.rmtree(self.base_folder_path)
            else:
                try:
                    os.removedirs(self.base_folder_path)
                except OSError:
                    pass

    def __repr__(self):
        return self.folder_path

    def __del__(self):
        self.close()


class KeywordFilter(logging.Filter):
    """
    A customized keyword filter that can be added to a log handler or a logger. Filters
    all the log messages, and if the message content contains the keyword, the log will
    not be printed.
    """

    def __init__(self, keyword=""):
        super().__init__(keyword)
        self.keyword = keyword

    def filter(self, record: logging.LogRecord):
        if self.keyword in record.msg:
            return False
        return True


class ModuleFilter(logging.Filter):
    """
    A customized module filter that can be added to a log handler or a logger. Filters
    all the log messages, and if the log was produced by a module with the specified
    module name, the log will not pass.
    """

    def __init__(self, module_name=""):
        super().__init__(module_name)
        self.module_name = module_name

    def filter(self, record: logging.LogRecord):
        if self.module_name in record.module:
            return False
        return True


class LevelFilter(logging.Filter):
    """
    Filter out the debug messages from the Jupyter logs. This is needed because the
    specialized logging functions, like code or output have smaller level than the
    DEBUG logging level (so that other than Jupyter handlers don't process them).
    """

    def __init__(self, level):
        super().__init__()
        self.level = logging.getLevelName(level)

    def filter(self, record: logging.LogRecord):
        if isinstance(self.level, int) and record.levelno == self.level:
            return False
        return True


logging.setLoggerClass(BasicLogger)
"""
These specialized logging levels are registered with the logging system, so they can be
retrieved by using for example logging.getLevelName('CODE').
"""
logging.addLevelName(LoggerLevel.OUTPUT.value, "OUTPUT")
logging.addLevelName(LoggerLevel.CODE.value, "CODE")


def import_logger_configuration(logger_config: dict, log_folder: LogFolder = None):
    """
    It imports the configuration of the loggers from a JSON data structure. This must be
    in the format described by `logging.config
    <https://docs.python.org/3/library/logging.config.html>`_ built-in module.

    It can also contain some additional settings:

    - :default_logger_directory: This is where a new log folder will be created for each
        execution. If it is set to None the system's temp folder is used.
    - :default_logger_cleanup: Specifies whether the log folders should be removed after
        execution or not.
    - :default_logger_labber_style: If this is true, it will create a log folder
        hierarchy in labber style at the specified **default_logger_directory**

    The logger list in the config file may also contain some additional settings:

    - :class: If this is specified, then the logger is of a custom class, not included
        in the built-in logging package. Similar to how the handlers are defined by the
        '()' key if they are custom handlers.
    - :active: If this is false, than the corresponding logger will not be imported.
        It's an easier way not to include a logger than to remove it from the config
        file, because then if the logger will be needed sometime, it doesn't require to
        re-write the config file, just change **active** from false to true.

    The configuration may also contain the starting log folder settings
    (:paramref:`log_folder`). Each time the logging configuration is imported, the log
    folder will be set up as it is specified. If this is not provided in the JSON
    structure, than the created log folder will use the default settings (which can also
    be specified in the configuration, as described above).


    :param logger_config: The JSON data structure from the logger_settings.json
                          configuration
    :param log_folder: The log folder to be used instead of the one specified in the
                       configuration file
    :return: List of the imported loggers. They are already loaded and configured.
    """
    if log_folder is None:
        if "log_folder" in logger_config:
            log_folder = LogFolder(**logger_config["log_folder"])
        else:
            log_folder = LogFolder()

    if "external_loggers" in logger_config:
        logger_config["loggers"].update(logger_config["external_loggers"])

    non_active_loggers = []
    for key, value in logger_config["loggers"].items():
        if "active" in value and value["active"] == 0:
            non_active_loggers.append(key)

    for key, value in logger_config["handlers"].items():
        # If we have a path append the current log folder onto it.
        file_path = value.get("file_path", None)
        if file_path is not None:
            value["file_path"] = os.path.join(log_folder.folder_path, file_path)

    for key in non_active_loggers:
        logger_config["loggers"].pop(key)

    dictConfig(logger_config)
    log_keys = set(logger_config["loggers"].keys())
    if "external_loggers" in logger_config:
        log_keys.difference_update(set(logger_config["external_loggers"].keys()))

    return CompositeLogger(list(log_keys), log_folder)


def get_logger_config(config_file=None, log_folder: LogFolder = None):
    """
    It imports the logger configuration from the provided JSON file. If this is not
    provided, then the current directory is searched for a logger_settings.json
    configuration file. If not found, then the default JSON file is read from
    qat/purr/logger_settings.json

    :param config_file: The path to the JSON file on the disk containing the logger
                        configuration
    :param log_folder: The log folder to be used instead of the one specified in the
                       configuration file
    :return: A DefaultLogger instance configured with the names of the imported loggers
    """

    if config_file is None:
        config_file = "logger_settings.json"

    potential_file = config_file
    if not os.path.isfile(potential_file):
        potential_file = os.path.join(os.getcwd(), config_file)
        if not os.path.isfile(potential_file):
            potential_file = os.path.join(os.path.dirname(__file__), config_file)

    if not os.path.isfile(potential_file):
        print(
            f"Log config file {config_file} doesn't exist and can't be found using "
            "default search patterns. Loading default configuration."
        )
        potential_file = os.path.join(os.path.dirname(__file__), "logger_settings.json")

    with open(potential_file, "r") as f:
        logger_config = json_load(f)
        return import_logger_configuration(logger_config, log_folder)


@atexit.register
def close_logger():
    """
    This method is executed upon exit, and it closes all the file handlers from the
    default loggers.

    This allows to remove the log folder after the execution is finished. This is needed
    because otherwise, the handlers would be closed after everything else, after the log
    folder is being removed.
    """
    def_logger = get_default_logger()
    if def_logger is not None:
        def_logger.close()


def load_object_from_log_folder(file_path: str):
    """
    Loads and deserializes an object from its JSON representation from the disk.

    :param file_path: The JSON file on the disk
    :return: The loaded object after deserialization
    """
    extension = Path(file_path).suffix
    if not extension == ".json":
        raise ValueError("Only JSON file paths are accepted!")
    with open(
        os.path.join(get_default_logger().log_folder.folder_path, file_path), "r"
    ) as f:
        obj = json_load(f)
    get_default_logger().debug(f"Object loaded: {str(obj)}")
    return obj


def save_object_to_log_folder(obj, sub_folder_path: str):
    """
    Serializes the specified object.
    """
    extension = Path(sub_folder_path).suffix
    if not extension == ".json":
        raise ValueError("Only JSON file paths are accepted!")
    sub_folder_path = os.path.join(
        get_default_logger().log_folder.folder_path, sub_folder_path
    )
    dir_name = os.path.dirname(sub_folder_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(sub_folder_path, "w") as f:
        json_dump(obj, f, indent=4, ensure_ascii=False)

    return sub_folder_path


_default_logging_instance = None


def get_default_logger():
    """
    Initializes the global logger or fetches one if it already exists.
    """
    global _default_logging_instance
    if _default_logging_instance is None:
        _default_logging_instance = get_logger_config()

        # Do a very hacky check about whether we're in a testing environment and force
        # clean-up if we are.
        try:
            import traceback

            stack = traceback.extract_stack()
            is_test_env = any(
                val.filename is not None and val.filename.endswith(f"unittest\\loader.py")
                for val in stack
            )
            if is_test_env:
                _default_logging_instance.log_folder.needs_cleanup = True
        except:
            _default_logging_instance.warning("Test environment detection failed.")

    return _default_logging_instance
