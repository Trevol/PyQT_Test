import sys

def my_excepthook(type, value, tback):
    # log the exception here
    print(type, value, tback)
    # then call the default handler
    sys.__excepthook__(type, value, tback)

sys.excepthook = my_excepthook