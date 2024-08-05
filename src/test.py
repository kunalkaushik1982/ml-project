import sys
from exception import CustomException

try:
    # Some code that may raise an exception
    1 / 0
except Exception as e:
    raise CustomException(str(e), sys)