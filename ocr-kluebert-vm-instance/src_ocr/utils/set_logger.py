import logging


class Log:
    def __init__(self):
        self.log = logging.getLogger()
        self.log.root.handlers = []
        self.log.propagate = True
        self.formatter = logging.Formatter("%(asctime)s;[%(levelname)s];%(message)s",
                                           "%Y-%m-%d %H:%M:%S")
        self.levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL}

    def set_stream_handler(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.formatter)
        self.log.addHandler(stream_handler)
        return self.log

    def set_file_handler(self,file_path, filename, mode='a'):
        file_name = file_path + filename
        file_handler = logging.FileHandler(file_name, mode=mode)
        file_handler.setFormatter(self.formatter)
        self.log.addHandler(file_handler)
        return self.log

    def set_log(self, log_path, filename="train.log", level="DEBUG"):
        self.set_stream_handler()
        self.set_file_handler(log_path, filename)
        self.log.setLevel(self.levels[level])
        return self.log

    def remove_handler(self):
        self.log.root.handlers = []
