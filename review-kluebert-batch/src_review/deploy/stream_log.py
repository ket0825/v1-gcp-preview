from utils.set_logger import Log

class StreamLog(Log):    
    def __init__(self):
        super().__init__()
        
    def set_stream_handler(self):
        super().set_stream_handler()
    
    def set_log(self, level="DEBUG"):
        self.set_stream_handler()
        self.log.setLevel(self.levels[level])
        return self.log