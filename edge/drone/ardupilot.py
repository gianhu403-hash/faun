import asyncio
import math
from edge.drone.base import DroneInterface, GpsPosition, Photo

class ArduPilotDrone(DroneInterface):
    def __init__(self, connection: str = "udp:127.0.0.1:14550"):
        self.connection = connection
        self._mav = None
        self._home_lat = None
        self._home_lon = None

    def _get_mav(self):
        if self._mav is None:
            from pymavlink import mavutil
            self._mav = mavutil.mavlink_connection(self.connection)
            self._mav.wait_heartbeat()
        return self._mav

    async def takeoff(self, altitude: float = 30.0) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_takeoff, altitude)

    def _sync_takeoff(self, altitude: float):
        from pymavlink import mavutil
        mav = self._get_mav()

        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
        if msg:
            self._home_lat = msg.lat / 1e7
            self._home_lon = msg.lon / 1e7

        mav.set_mode_apm('GUIDED')
        import time; time.sleep(1)
        mav.arducopter_arm()
        mav.motors_armed_wait()
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )
        import time; time.sleep(5)

    async def fly_to(self, lat: float, lon: float):
        from pymavlink import mavutil
        mav = self._get_mav()

        mav.mav.set_position_target_global_int_send(
            0, mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,
            int(lat * 1e7), int(lon * 1e7), 30,
            0, 0, 0, 0, 0, 0, 0, 0
        )

        while True:
            await asyncio.sleep(0.5)
            loop = asyncio.get_event_loop()
            msg = await loop.run_in_executor(
                None,
                lambda: mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            )
            if not msg:
                continue

            curr_lat = msg.lat / 1e7
            curr_lon = msg.lon / 1e7
            yield GpsPosition(lat=curr_lat, lon=curr_lon)

            dist = math.sqrt((curr_lat - lat)**2 + (curr_lon - lon)**2) * 111000
            if dist < 3.0:
                break

    async def capture_photo(self) -> Photo:
        from pymavlink import mavutil
        mav = self._get_mav()

        # Trigger shutter
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_DIGICAM_CONTROL,
            0, 0, 0, 0, 1, 0, 0, 0
        )
        await asyncio.sleep(2)

        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=3)
        lat = msg.lat / 1e7 if msg else 0.0
        lon = msg.lon / 1e7 if msg else 0.0

        return Photo(data=b"", lat=lat, lon=lon)

    async def return_home(self) -> None:
        from pymavlink import mavutil
        mav = self._get_mav()
        mav.set_mode_apm('RTL')
        await asyncio.sleep(10)
