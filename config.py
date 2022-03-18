import configparser

config = configparser.ConfigParser()

config.optionxform = str
config.read('./config.ini')

default_config = config['DEFAULT']

