import abc
from queue import Queue
from threading import Event


class InterruptError(RuntimeError): ...


class Interrupt(abc.ABC):
    @abc.abstractmethod
    def if_triggered(self, metadata={}, throw=False):
        """
        Capture context if triggered
        """
        ...

    @abc.abstractmethod
    def trigger(self):
        """
        Set the triggered status
        """
        ...


class NullInterrupt(Interrupt):
    def if_triggered(self, metadata={}, throw=False): ...

    def trigger(self): ...


class BasicInterrupt(Interrupt):
    """
    Interrupt supporting multi-threading and multi-process concurrency
    """

    def __init__(self, event=Event(), queue=Queue()):
        self._event = event
        self._queue = queue
        self._metadata = {}

    def if_triggered(self, metadata={}, throw=False):
        self._metadata.update(metadata)
        if self._event.is_set():
            response = {"interrupted": True, "metadata": self._metadata}
            self._queue.put(response)
            if throw:
                raise InterruptError(f"Interrupt triggered, {self._metadata}")

    def trigger(self):
        self._event.set()

    @property
    def event(self):
        return self._event

    @property
    def queue(self):
        return self._queue
