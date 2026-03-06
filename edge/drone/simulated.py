import asyncio
import os
from pathlib import Path
from edge.drone.base import DroneInterface, GpsPosition, Photo

DEMO_PHOTOS_DIR = Path(__file__).parent.parent.parent / "demo" / "photos"

class SimulatedDrone(DroneInterface):

    def __init__(self, home_lat: float = 55.750, home_lon: float = 37.610):
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.current_lat = home_lat
        self.current_lon = home_lon

    async def takeoff(self) -> None:
        await asyncio.sleep(1.0)

    async def fly_to(self, lat: float, lon: float):
        steps = 8
        for i in range(1, steps + 1):
            self.current_lat = self.home_lat + (lat - self.home_lat) * i / steps
            self.current_lon = self.home_lon + (lon - self.home_lon) * i / steps
            yield GpsPosition(lat=self.current_lat, lon=self.current_lon)
            await asyncio.sleep(0.4)

    async def capture_photo(self) -> Photo:
        await asyncio.sleep(0.5)

        photos = list(DEMO_PHOTOS_DIR.glob("*.jpg"))
        if not photos:
            data = bytes([
                0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46,
                0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
                0xff, 0xdb, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06,
                0xff, 0xd9,
            ])
        else:
            data = photos[0].read_bytes()

        return Photo(data=data, lat=self.current_lat, lon=self.current_lon)

    async def return_home(self) -> None:
        await asyncio.sleep(1.0)
        self.current_lat = self.home_lat
        self.current_lon = self.home_lon
