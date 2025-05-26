from abc import ABC
import threading
import datetime
import queue
import inspect
import os
from colorama import Fore
from colorama import Style
import traceback

class LogLevel(ABC):
    Number = 0
    Word = ""

class CRITICAL(LogLevel):
    Number = 40
    Word = "CRITICAL"


class ERROR(LogLevel):
    Number = 30
    Word = "ERROR"


class WARNING(LogLevel):
    Number = 20
    Word = "WARNING"


class INFO(LogLevel):
    Number = 10
    Word = "INFO"


class DEBUG(LogLevel):
    Number = 0
    Word = "DEBUG"


class Logging:
    def _LoggingThread(self):
        while True:
            LogItem = self.LogQueue.get()
            with open(os.path.join(self.LogPath, self.LogFileName), "a") as F:
                F.write(LogItem + "\n")

    def __init__(
        self,
        LogToFile=True,
        LogToConsole=True,
        LogPath="logs/",
        AutoGenerateFileName=True,
        ManualFileName="ProcessLog",
        LogConsoleLevel=INFO,
        LogFileLevel=DEBUG,
        LogMessagePrefix="[%B %d,%Y %H:%M:%S:%f]",
        ShowTime: bool = True,
    ) -> None:
        self.LogToFile = LogToFile
        self.LogToConsole = LogToConsole
        self.LogPath = LogPath
        self.AutoGenerateFileName = AutoGenerateFileName
        self.LogFileName = ManualFileName + ".txt"
        if self.AutoGenerateFileName:
            self.LogFileName = (
                f"{ManualFileName}:"
                + datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S:%f")
                + ".txt"
            )
        self.LogConsoleLevel = LogConsoleLevel
        self.LogFileLevel = LogFileLevel
        self.LogMessagePrefix = LogMessagePrefix
        self.LogQueue = queue.Queue()
        self.ShowTime = ShowTime
        if self.LogToFile:
            self.LogThreadReffrence = threading.Thread(
                target=self._LoggingThread, args=(), daemon=True
            )
            self.LogThreadReffrence.start()

    def _AddToLogging(self, Message: str, LogLevel: LogLevel, Color, Caller, LocalLogLevel: LogLevel):
        if LocalLogLevel is not None:
            if LogLevel.Number < LocalLogLevel.Number:
                return

        if Caller == None:
            Caller = inspect.stack()[2]
            CallerFile = Caller.filename.replace("\\", "/")
            Caller = f"{CallerFile.split('/')[-1]}/{Caller.function}:{Caller.lineno}"
        Time = ""
        if self.ShowTime == True:
            Time = datetime.datetime.now().strftime(self.LogMessagePrefix) + " "
        FormattedMessage = (
            f"{Color}{Time}{LogLevel.Word} [{Caller}] : {Message}{Style.RESET_ALL}"
        )
        if self.LogToConsole:
            if LogLevel.Number >= self.LogConsoleLevel.Number:
                print(FormattedMessage)
        if self.LogToFile:
            if LogLevel.Number >= self.LogFileLevel.Number:
                self.LogQueue.put(FormattedMessage)

    def Critical(self, Message: str, Caller=None, LocalLogLevel: LogLevel = None):
        self._AddToLogging(
            Message=Message,
            LogLevel=CRITICAL,
            Color=Fore.RED + Style.BRIGHT,
            Caller=Caller,
            LocalLogLevel=LocalLogLevel,
        )

    def Error(self, Message: str, Caller=None, LocalLogLevel: LogLevel = None):
        self._AddToLogging(
            Message=Message, LogLevel=ERROR, Color=Fore.RED, Caller=Caller,
            LocalLogLevel=LocalLogLevel,
        )

    def Warning(self, Message: str, Caller=None, LocalLogLevel: LogLevel = None):
        self._AddToLogging(
            Message=Message, LogLevel=WARNING, Color=Fore.YELLOW, Caller=Caller,
            LocalLogLevel=LocalLogLevel,
        )

    def Info(self, Message: str, Caller=None, LocalLogLevel: LogLevel = None):
        self._AddToLogging(
            Message=Message, LogLevel=INFO, Color=Fore.WHITE, Caller=Caller,
            LocalLogLevel=LocalLogLevel,
        )

    def Debug(self, Message: str, Caller=None, LocalLogLevel: LogLevel = None):
        self._AddToLogging(
            Message=Message, LogLevel=DEBUG, Color=Fore.WHITE + Style.DIM, Caller=Caller,
            LocalLogLevel=LocalLogLevel,
        )

    def Format_Exception(self, E: BaseException) -> str:
        return ' '.join(traceback.format_exception(E))


logger: Logging = Logging(LogToFile=False, ShowTime=False)