import logging

def set_logger(log_path):
    """设置日志记录器，将信息记录到终端和文件`log_path`中。

    :param log_path: 日志记录的文件路径
    :type log_path: str
    """
    logger = logging.getLogger()  # 创建一个logger对象
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO

    if not logger.handlers:
        # 如果logger对象没有处理器，则创建处理器并添加到logger对象中

        # 日志记录到文件
        file_handler = logging.FileHandler(log_path)  # 创建一个FileHandler对象，将日志记录到文件
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))  # 设置日志格式
        logger.addHandler(file_handler)  # 将file_handler添加到logger对象中

        # 日志记录到终端
        stream_handler = logging.StreamHandler()  # 创建一个StreamHandler对象，将日志记录到终端
        stream_handler.setFormatter(logging.Formatter('%(message)s'))  # 设置日志格式
        logger.addHandler(stream_handler)  # 将stream_handler添加到logger对象中
