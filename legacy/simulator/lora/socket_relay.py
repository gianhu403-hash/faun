import asyncio
import json
from dataclasses import asdict

class LoraRelay:
    """Edge side — sends packet over simulated LoRa."""

    def __init__(self, host: str = "localhost", port: int = 9000):
        self.host = host
        self.port = port

    async def send(self, packet: dict) -> None:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        writer.write(json.dumps(packet).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()


class LoraGateway:
    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port
        self._on_packet = None

    def on_packet(self, callback):
        """Register callback for incoming packets."""
        self._on_packet = callback
        return callback

    async def start(self) -> None:
        server = await asyncio.start_server(
            self._handle, self.host, self.port
        )
        async with server:
            await server.serve_forever()

    async def _handle(self, reader, writer):
        data = await reader.readline()
        writer.close()
        if data and self._on_packet:
            packet = json.loads(data.decode())
            await self._on_packet(packet)
