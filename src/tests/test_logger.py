# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import datetime
import inspect
import json
import logging
import os
import re
import tempfile
import unittest
from tempfile import gettempdir

import qat.purr.utils.logger as logger
from qat.purr.utils.serializer import json_load

info_msg_pattern = \
    r"^\[INFO\] \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \(\w+.\w+:\d+\) - (.+)\n"


class LoggingCleanup:
    def __init__(self, tempdir, comp_logger, _logger, handler):
        self.tempdir = tempdir
        self.comp_logger: logger.CompositeLogger = comp_logger
        self.logger = _logger
        self.handler = handler

    def __enter__(self):
        return self.tempdir.name, self.comp_logger.logs_path, self.logger, self.handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.comp_logger.close()
        self.tempdir.cleanup()


class TestFileLogger:
    def create_logger_instance(self):
        method_name = inspect.stack()[1].function

        temp_dir = tempfile.TemporaryDirectory()
        def_logger = logger.CompositeLogger(
            _log_folder=logger.LogFolder(temp_dir.name, prefix=method_name)
        )
        file_logger = logging.getLogger(f"test.{method_name}.file")
        file_handler = logger.FileLoggerHandler(
            os.path.join(def_logger.logs_path, f"{method_name}.txt")
        )
        file_logger.addHandler(file_handler)
        file_logger.setLevel(logging.INFO)
        def_logger.add_loggers([file_logger])
        return temp_dir, def_logger, file_logger, file_handler

    def test_file_is_created_in_log_folder(self):
        with LoggingCleanup(*self.create_logger_instance()
                           ) as (tempdir, logs_path, file_logger, file_handler):
            assert os.path.dirname(file_handler.baseFilename), logs_path

    def test_file_is_created_delayed(self):
        with LoggingCleanup(*self.create_logger_instance()
                           ) as (tempdir, logs_path, file_logger, file_handler):
            assert not os.path.exists(file_handler.baseFilename)
            file_logger.info("Hello world!")
            assert os.path.exists(file_handler.baseFilename)
            file_handler.close()

    def test_file_default_info_format_is_used(self):
        with LoggingCleanup(*self.create_logger_instance()
                           ) as (tempdir, logs_path, file_logger, file_handler):
            msg = "Hello world!"
            file_logger.info(msg)
            with open(file_handler.baseFilename, "r") as f:
                text = f.read()
                regex_match = re.match(info_msg_pattern, text)
                assert regex_match
                assert regex_match.group(1) == msg
            file_handler.close()

    class TestFileLoggerHandler(logger.FileLoggerHandler):
        initial_text = "Initial text"

        def create_initial_file(self):
            self.stream.write(self.initial_text + '\n')
            self.flush()

    def test_file_initial_text_is_written_if_specified(self):
        with LoggingCleanup(*self.create_logger_instance()
                           ) as (tempdir, logs_path, file_logger, file_handler):
            file_logger.removeHandler(file_handler)
            file_handler.close()
            file_handler = self.TestFileLoggerHandler(
                os.path.join(
                    logs_path, "initial_test_file_initial_text_is_written_if_specified.txt"
                )
            )
            file_logger.addHandler(file_handler)
            msg = "Actual message"

            file_logger.info(msg)

            with open(file_handler.baseFilename, "r") as f:
                text = f.read().split('\n')
                assert text[0] == self.TestFileLoggerHandler.initial_text
                regex_match = re.match(info_msg_pattern, text[1] + '\n')
                assert regex_match
                assert regex_match.group(1) == msg


class TestJsonLogger:
    def create_logger_instance(self):
        method_name = inspect.stack()[1].function

        temp_folder = tempfile.TemporaryDirectory()
        def_logger = logger.CompositeLogger()
        def_logger.log_folder = logger.LogFolder(temp_folder.name, prefix=method_name)

        json_logger = logging.getLogger(f"test.{method_name}.json")
        json_handler = logger.JsonLoggerHandler(
            os.path.join(temp_folder.name, f"{method_name}.json")
        )
        json_logger.addHandler(json_handler)
        json_logger.setLevel(logging.INFO)
        def_logger.loggers = [json_logger]
        return temp_folder, def_logger, json_logger, json_handler

    def test_json_valid(self):
        with LoggingCleanup(*self.create_logger_instance()
                           ) as (tempdir, logs_path, json_logger, json_handler):
            msg = "Test message"
            json_logger.info(msg)
            json_logger.info(msg)
            assert os.path.exists(json_handler.baseFilename)
            f = open(json_handler.baseFilename, "r")
            try:
                json_load(f)
            except json.JSONDecodeError:
                raise ValueError("Not a valid JSON document!")
            finally:
                f.close()
            json_handler.close()

    def test_json_info_log(self):
        with LoggingCleanup(*self.create_logger_instance()
                           ) as (tempdir, logs_path, json_logger, json_handler):
            msg = "Test message"
            json_logger.info(msg)
            with open(json_handler.baseFilename, "r") as f:
                res = json_load(f)
                assert res["entries"][0]["level"] == "INFO"
                assert res["entries"][0]["message"] == msg
                json_handler.close()


class LogFolderTests(unittest.TestCase):
    def test_temp_folder_at_system_temp_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            log_folder = logger.LogFolder(base_folder_path=tempdir)
            assert os.path.split(log_folder.folder_path)[0] == gettempdir()
            assert os.path.exists(log_folder.folder_path)

    def test_temp_folder_at_specific_nested_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            specific_nested_dir = os.path.join('test_logs', self._testMethodName)
            log_folder = logger.LogFolder(
                base_folder_path=tempdir, folder_name=specific_nested_dir
            )
            assert log_folder.folder_path == os.path.join(tempdir, specific_nested_dir)

    def test_specified_nested_folder_at_system_temp_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            specific_nested_log_folder = os.path.join('test_logs', self._testMethodName)
            log_folder = logger.LogFolder(
                base_folder_path=tempdir, folder_name=specific_nested_log_folder
            )
            assert log_folder.folder_path == os.path.join(
                tempdir, specific_nested_log_folder
            )
            assert os.path.exists(log_folder.folder_path)

    def test_temp_folder_with_prefix_and_suffix(self):
        with tempfile.TemporaryDirectory() as tempdir:
            prefix = 'prefix'
            suffix = 'suffix'
            log_folder = logger.LogFolder(
                base_folder_path=tempdir, prefix=prefix, suffix=suffix
            )
            assert re.match(
                f"{prefix}_.*_{suffix}", os.path.basename(log_folder.folder_path)
            )
            assert os.path.exists(log_folder.folder_path)

    def test_specified_folder_name_with_prefix_and_suffix(self):
        with tempfile.TemporaryDirectory() as tempdir:
            prefix = 'prefix'
            suffix = 'suffix'
            name = self._testMethodName
            log_folder = logger.LogFolder(
                base_folder_path=tempdir, folder_name=name, prefix=prefix, suffix=suffix
            )
            assert f"{prefix}_{name}_{suffix}" == os.path.basename(log_folder.folder_path)
            assert os.path.exists(log_folder.folder_path)

    def test_temp_folder_with_cleanup(self):
        with tempfile.TemporaryDirectory() as tempdir:
            log_folder = logger.LogFolder(tempdir, cleanup=True)
            assert os.path.exists(log_folder.folder_path)
            log_folder.close()
            assert not os.path.exists(log_folder.folder_path)

    def test_temp_folder_without_cleanup(self):
        with tempfile.TemporaryDirectory() as tempdir:
            log_folder = logger.LogFolder(tempdir, cleanup=False)
            assert os.path.exists(log_folder.folder_path)
            empty_file = open(os.path.join(log_folder.folder_path, "empty.txt"), "a")
            empty_file.close()
            log_folder.close()
            assert os.path.exists(log_folder.folder_path)

    def test_temp_folder_without_cleanup_empty(self):
        with tempfile.TemporaryDirectory() as tempdir:
            log_folder = logger.LogFolder(tempdir, cleanup=False)
            assert os.path.exists(log_folder.folder_path)
            log_folder.close()
            assert not os.path.exists(log_folder.folder_path)

    def test_str_repr_is_folder_path(self):
        log_folder = logger.LogFolder(base_folder_path="something/else")
        assert str(log_folder) == log_folder.folder_path

    def assert_sub_folder_path(
        self,
        current_time,
        main_folder,
        sub_folder_path,
        sub_folder_name='',
        sub_folder_index=-1
    ):
        assert os.path.exists(sub_folder_path)
        (main_folder_path, time_folder) = os.path.split(sub_folder_path)
        if sub_folder_index >= 0:
            time_folder = time_folder[:-(1 + len(str(sub_folder_index)))]
        if not sub_folder_name == '':
            (head, tail) = os.path.split(sub_folder_name)
            time_folder = time_folder[(1 + len(tail)):]
            while not head == '':
                head = os.path.split(head)[0]
                main_folder_path = os.path.split(main_folder_path)[0]
        (main_folder_path, month_day_folder) = os.path.split(main_folder_path)
        day = month_day_folder[-2:]
        (main_folder_path, month_folder) = os.path.split(main_folder_path)
        (main_folder_path, year_folder) = os.path.split(main_folder_path)
        assert main_folder_path == main_folder
        sub_folder_created_at = datetime.datetime.strptime(
            "-".join([year_folder, month_folder, day, time_folder]), '%Y-%m-%d-%H.%M.%S'
        )
        assert (current_time - sub_folder_created_at).total_seconds() < 5

    def test_empty_sub_folder_name(self):
        with tempfile.TemporaryDirectory() as tempdir:
            log_folder = logger.LogFolder(tempdir, labber_style=True)
            now = datetime.datetime.now()
            self.assert_sub_folder_path(now, tempdir, log_folder.folder_path)

    def test_nested_sub_folder_name(self):
        with tempfile.TemporaryDirectory() as tempdir:
            sub_folder_name = 'test1/test2'
            now = datetime.datetime.now()
            log_folder = logger.LogFolder(
                os.path.join(tempdir, '.logs'),
                labber_style=True,
                folder_name=sub_folder_name
            )
            self.assert_sub_folder_path(
                now,
                os.path.abspath(os.path.join(tempdir, '.logs')),
                log_folder.folder_path,
                sub_folder_name=sub_folder_name
            )

    def test_existing_sub_folder_name(self):
        with tempfile.TemporaryDirectory() as tempdir:
            now = datetime.datetime.now()
            log_folder = logger.LogFolder(
                os.path.join(tempdir, '.logs'), labber_style=True, folder_name='test'
            )
            log_folder_2 = logger.LogFolder(
                os.path.join(tempdir, '.logs'), labber_style=True, folder_name='test'
            )
            self.assert_sub_folder_path(
                now,
                os.path.abspath(os.path.join(tempdir, '.logs')),
                log_folder.folder_path,
                'test'
            )
            self.assert_sub_folder_path(
                now,
                os.path.abspath(os.path.join(tempdir, '.logs')),
                log_folder_2.folder_path,
                'test',
                sub_folder_index=0
            )
