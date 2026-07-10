from __future__ import annotations

import importlib.util
import os
import sys


# Resolve the protocols.py path relative to this conftest
_conftest_dir = os.path.dirname(os.path.abspath(__file__))
_kafka_pkg_dir = os.path.normpath(os.path.join(_conftest_dir, "..", ".."))
_protocols_path = os.path.join(_kafka_pkg_dir, "protocols.py")

# Load the protocols module BEFORE any test files import from the kafka
# package, so that consumer.py / producer.py can find the missing
# ConfluentConsumer / ConfluentProducer exports.
_spec = importlib.util.spec_from_file_location(
    "src.platform.messaging.kafka.protocols", _protocols_path
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["src.platform.messaging.kafka.protocols"] = _mod
_spec.loader.exec_module(_mod)

# Add the attributes that consumer.py and producer.py expect from protocols.
# Use real classes with no-op __init__ so the constructor call in
# KafkaConsumerClient.__init__ / KafkaProducerClient.__init__ does not fail.
class _ConfluentConsumer:
    def __init__(self, *args, **kwargs) -> None: ...

class _ConfluentProducer:
    def __init__(self, *args, **kwargs) -> None: ...

_mod.ConfluentConsumer = _ConfluentConsumer
_mod.ConfluentProducer = _ConfluentProducer
