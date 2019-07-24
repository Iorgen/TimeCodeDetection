import os
from core.controller import TimeCodeController
from app import create_app
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
TimeCodeController()
app = create_app(os.getenv('FLASK_CONFIG') or 'default')

