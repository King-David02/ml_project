import sys

class CustomException(Exception):
    def __init__(self, error, error_detail: tuple):
        super().__init__(error)
        self.error = error
        _, _, tb = error_detail
        self.filename = tb.tb_frame.f_code.co_filename
        self.line_number = tb.tb_lineno

    def __str__(self):
        return f'The "Error" {self.error}, occurred in "File" {self.filename}, at line {self.line_number}'

# Example Usage
#try:
#    1 / 0  # This will cause a ZeroDivisionError
#except Exception as e:
#    raise CustomException(e, sys.exc_info())
