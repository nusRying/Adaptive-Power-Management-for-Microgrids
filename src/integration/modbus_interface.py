from __future__ import annotations

from dataclasses import dataclass

try:
    from pymodbus.client import ModbusTcpClient
except ImportError:  # pragma: no cover
    ModbusTcpClient = None


@dataclass
class ModbusConfig:
    host: str = "127.0.0.1"
    port: int = 502
    unit_id: int = 1
    battery_setpoint_register: int = 100
    grid_setpoint_register: int = 101


class ModbusDispatcher:
    """
    Minimal Modbus adapter to send EMS setpoints to converter/PLC registers.
    """

    def __init__(self, config: ModbusConfig):
        self.config = config
        self.client = None

    def connect(self) -> None:
        if ModbusTcpClient is None:
            raise RuntimeError(
                "pymodbus is not installed. Install dependencies before Modbus integration."
            )
        self.client = ModbusTcpClient(host=self.config.host, port=self.config.port)
        self.client.connect()

    def dispatch(self, battery_kw: float, grid_kw: float) -> None:
        if self.client is None:
            raise RuntimeError("Modbus client is not connected.")
        self.client.write_register(
            address=self.config.battery_setpoint_register,
            value=int(round(battery_kw)),
            slave=self.config.unit_id,
        )
        self.client.write_register(
            address=self.config.grid_setpoint_register,
            value=int(round(grid_kw)),
            slave=self.config.unit_id,
        )

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

